"""Utilities for benchmarking."""
import importlib
import json
import pickle
from copy import copy
from dataclasses import dataclass, field
from time import sleep
from timeit import default_timer as t_now
from typing import Any, Dict, List, Optional, Tuple, Union

import affine
import distributed
import numpy as np
import pystac.item
import xarray as xr
from dask.utils import format_bytes
from odc.geo import CRS
from odc.geo.xr import ODCExtension

import odc.stac

TimeSample = Tuple[float, float, float]
"""(t0, t_finished_submit, t_finished_compute)"""

# pylint: disable=too-many-instance-attributes,too-many-locals,too-many-arguments
# pylint: disable=import-outside-toplevel,import-error


@dataclass
class BenchmarkContext:
    """
    Benchmark Context Metadata.

    Normalized representation of the task being benchmarked and
    the environment it is benchmarked in.
    """

    #################################
    # Cluster stats
    cluster_info: Dict[str, Any] = field(repr=False, init=True, compare=False)
    """client.scheduler_info().copy()"""

    nworkers: int = field(init=False)
    """Number of workers in the cluster"""

    nthreads: int = field(init=False)
    """Number of threads in the cluster across all workers"""

    total_ram: int = field(init=False)
    """Total RAM across all workers of the cluster"""

    #################################
    # Data shape stats
    npix: int
    """Total number of output pixels across all bands/timeslices"""

    nbytes: int
    """Total number of output bytes across bands/timeslices"""

    dtype: str
    """Data type of a pixel, assumed to be the same across bands"""

    shape: Tuple[int, int, int, int]
    """Normalized shape in time,band,y,x order"""

    chunks: Tuple[int, int, int, int]
    """Normalized chunk size in time,band,y,x order"""

    #################################
    # Geo
    crs: str
    """Projection used for output"""

    transform: affine.Affine
    """Linear mapping from pixel coordinates to CRS coordinates"""

    #################################
    # Misc
    scenario: str = ""
    """Some human name for this benchmark data"""

    temporal_id: str = ""
    """Time period covered by data in human readable form"""

    method: str = field(compare=False, default="undefined")
    """Some human name for method, (stackstac, odc-stac, rio-xarray)"""

    extras: Dict[str, Any] = field(compare=False, default_factory=dict)
    """Any other parameters to capture"""

    def __post_init__(self):
        """Extract stats from cluster_info."""
        self.nworkers = len(self.cluster_info["workers"])
        self.nthreads = sum(
            w["nthreads"] for w in self.cluster_info["workers"].values()
        )
        self.total_ram = sum(
            w["memory_limit"] for w in self.cluster_info["workers"].values()
        )

    @property
    def data_signature(self) -> str:
        """Render textual representation of data shape and type."""
        data_dims = ".".join(map(str, self.shape))
        return f"{data_dims}.{self.dtype}"

    @property
    def chunk_signature(self) -> str:
        """Render textual representation of chunk shapes."""
        return ".".join(map(str, self.chunks))

    def render_txt(self, col_width: int = 10) -> str:
        """
        Render textual representation for human consumption.

        :param col_width: Left column width in characters, defaults to 10
        :return: Multiline string representation of self
        """
        nw = col_width

        transorm_txt = f"\n{'':{nw}}".join(str(self.transform).split("\n")[:2])
        transorm_txt = transorm_txt.replace(".00,", ",")
        transorm_txt = transorm_txt.replace(".00|", "|")

        return f"""
{"method":{nw}}: {self.method}
{"Scenario":{nw}}: {self.scenario}
{"T.slice":{nw}}: {self.temporal_id}
{"Data":{nw}}: {self.data_signature},  {format_bytes(self.nbytes)}
{"Chunks":{nw}}: {self.chunk_signature} (T.B.Y.X)
{"GEO":{nw}}: {self.crs}
{"":{nw}}{transorm_txt}
{"Cluster":{nw}}: {self.nworkers} workers, {self.nthreads} threads, {format_bytes(self.total_ram)}
""".strip()

    def render_timing_info(
        self, times: Tuple[float, float, float], col_width: int = 10
    ) -> str:
        """
        Render textual representation of timing result.

        :param times: (t0, t_submit_done, t_done)
        :param col_width: Left column width in characters, defaults to 10
        :return: Multiline string for human consumption
        """
        t0, t1, t2 = times
        t_submit = t1 - t0
        t_elapsed = t2 - t0
        nw = col_width
        return f"""
{"T.Elapsed":{nw}}: {t_elapsed:8.3f} seconds
{"T.Submit":{nw}}: {t_submit:8.3f} seconds
{"Throughput":{nw}}: {self.npix/(t_elapsed*1e+6):8.3f} Mpx/second (overall)
{"":{nw}}| {self.npix/(self.nthreads*t_elapsed*1e+6):8.3f} Mpx/second (per thread)
""".strip()

    @property
    def resolution(self):
        """Extract resolution."""
        sx, _, _, _, sy, *_ = self.transform
        return min(abs(v) for v in [sx, sy])

    def to_pandas_dict(self) -> Dict[str, Any]:
        """Extract parts one would need for analysis of results."""
        return dict(
            method=self.method,
            scenario=self.scenario,
            data=self.data_signature,
            chunks=self.chunk_signature,
            chunks_x=self.chunks[2],
            chunks_y=self.chunks[3],
            resolution=self.resolution,
            crs=self.crs,
            npix=self.npix,
            nbytes=self.nbytes,
            nthreads=self.nthreads,
            total_ram=self.total_ram,
        )


def collect_context_info(
    client: distributed.Client, xx: Union[xr.DataArray, xr.Dataset], **kw
) -> BenchmarkContext:
    """
    Assemble :class:`~odc.stac.bench.BenchmarkContext` metadata.

    :param client: Dask distributed client
    :param xx: :class:`~xarray.DataArray` or :class:`~xarray.Dataset` that will be benchmarked
    :param kw: Passed on to :class:`~odc.stac.bench.BenchmarkContext` constructor
    :raises ValueError: If ``xx`` is not of the appropriate type
    :return: Populated :class:`~odc.stac.bench.BenchmarkContext` object
    """
    if isinstance(xx, xr.DataArray):
        npix = xx.data.size
        dtype = xx.data.dtype
        nbytes = npix * dtype.itemsize
        _band = getattr(xx, "band", None)
        nb = _band.shape[0] if _band is not None and _band.ndim > 0 else 1
        assert xx.chunks is not None
        _chunks = {k: max(v) for k, v in zip(xx.dims, xx.chunks)}
    elif isinstance(xx, xr.Dataset):
        npix = sum(b.data.size for b in xx.data_vars.values())
        nbytes = sum(b.data.dtype.itemsize * b.data.size for b in xx.data_vars.values())
        dtype, *_ = {b.dtype for b in xx.data_vars.values()}
        nb = len(xx.data_vars)
        sample_band, *_ = xx.data_vars.values()
        assert sample_band.chunks is not None
        _chunks = {k: max(v) for k, v in zip(sample_band.dims, sample_band.chunks)}
    else:
        raise ValueError("Expect one of `xarray.{DataArray,Dataset}` on input")

    assert isinstance(xx.odc, ODCExtension)
    geobox = xx.odc.geobox
    assert geobox is not None
    yx_dims = geobox.dimensions
    ny, nx = (xx[dim].shape[0] for dim in yx_dims)

    time = getattr(xx, "time", None)
    if time is None:
        nt = 1
        temporal_id = "-"
    else:
        time = time.dt.strftime("%Y-%m-%d")
        if time.ndim == 0:
            nt = 1
            temporal_id = time.item()
        else:
            nt = time.shape[0]
            if nt == 1:
                temporal_id = time.data[0]
            else:
                temporal_id = f"{time.data[0]}__{time.data[-1]}"

    ct, cb, cy, cx = (_chunks.get(k, 1) for k in ["time", "band", *yx_dims])
    chunks = (ct, cb, cy, cx)
    geobox = xx.odc.geobox
    if geobox is None or geobox.crs is None:
        raise ValueError("Can't find GEO info")
    crs = f"epsg:{geobox.crs.epsg}"
    transform = geobox.transform

    return BenchmarkContext(
        client.scheduler_info().copy(),
        npix=npix,
        nbytes=nbytes,
        dtype=str(dtype),
        shape=(nt, nb, ny, nx),
        chunks=chunks,
        crs=crs,
        transform=transform,
        temporal_id=temporal_id,
        **kw,
    )


@dataclass
class BenchLoadParams:
    """Per experiment configuration."""

    scenario: str = ""
    """Name for this scenario"""

    method: str = "odc-stac"
    """Method to use for loading: ``odc-stac|stackstac``"""

    chunks: Tuple[int, int] = (2048, 2048)
    """Chunk size in pixels in ``Y, X`` order"""

    bands: Optional[Tuple[str, ...]] = None
    """Bands to load, defaults to All"""

    resolution: Optional[float] = None
    """Resolution, leave as ``None`` for native"""

    crs: Optional[str] = None
    """Projection to use, leave as ``None`` for native"""

    resampling: Optional[str] = None
    """Set resampling method when reprojecting at load"""

    patch_url: Any = None
    """Accepts ``planetary_computer.sign``"""

    extra: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    """Extra params per ``method``"""

    def with_method(self, method: str) -> "BenchLoadParams":
        """Replace method field only."""
        other = copy(self)
        other.method = method
        return other

    @property
    def epsg(self) -> Optional[int]:
        """Return EPSG code of crs if it was configured, else ``None``."""
        if self.crs is None:
            return None
        return CRS(self.crs).epsg

    @property
    def chunks_as_dict(self) -> Dict[str, float]:
        """Return chunks in dictionary form."""
        return {"y": self.chunks[0], "x": self.chunks[1]}

    def to_json(self, indent=2) -> str:
        """Convert to JSON string."""
        data = copy(self.__dict__)
        if self.patch_url is not None:
            data["patch_url"] = f"{self.patch_url.__module__}.{self.patch_url.__name__}"
        return json.dumps(data, indent=indent)

    @staticmethod
    def from_json(json_text: str) -> "BenchLoadParams":
        """Construct from JSON string."""
        src = json.loads(json_text)

        # convert arrays to tuples
        for k in ("bands", "chunks"):
            v = src.get(k, None)
            if v is not None:
                src[k] = tuple(v)

        cfg = BenchLoadParams(**src)
        if isinstance(cfg.patch_url, str):
            cfg.patch_url = _method_from_string(cfg.patch_url)
        return cfg

    def compute_args(self, method: str = "") -> Dict[str, Any]:
        """Translate into call arguments for a given method."""
        if method == "":
            method = self.method

        extra = dict(**self.extra.get(method, {}))

        if method == "odc-stac":
            return _trim_dict(
                {
                    "chunks": self.chunks_as_dict,
                    "crs": self.crs,
                    "resolution": self.resolution,
                    "patch_url": self.patch_url,
                    "bands": self.bands,
                    "resampling": self.resampling,
                    **extra,
                }
            )
        if method == "stackstac":
            from rasterio.enums import Resampling

            resampling = None
            if self.resampling is not None:
                resampling = Resampling[self.resampling]

            assets = None
            if self.bands is not None:
                # translate to list, stackstac doesn't like tuple
                assets = list(self.bands)

            extra.setdefault("dtype", "uint16")
            extra.setdefault("fill_value", 0)
            extra.setdefault("xy_coords", "center")

            return _trim_dict(
                {
                    "chunksize": self.chunks[0],
                    "epsg": self.epsg,
                    "resolution": self.resolution,
                    "assets": assets,
                    "resampling": resampling,
                    **extra,
                }
            )

        return {}


def _default_nodata(dtype):
    if dtype.kind == "f":
        return float("nan")
    return 0


def load_from_json(geojson, params: BenchLoadParams, **kw):
    """
    Turn passed in geojson into a Dask array.

    :param geojson: GeoJSON FeatureCollection
    :param params: data loading configuration
    :param kw: passed on to underlying data load function
    """
    all_items = [pystac.item.Item.from_dict(f) for f in geojson["features"]]

    opts = params.compute_args()
    opts.update(**kw)

    if params.method == "odc-stac":
        xx = odc.stac.load(all_items, **opts)
    elif params.method == "stackstac":
        import stackstac

        patch_url = params.patch_url
        if patch_url is None:
            patch_url = lambda x: x

        _items = [patch_url(item).to_dict() for item in all_items]
        xx = stackstac.stack(_items, **opts)
        if np.unique(xx.time.data).shape != xx.time.shape:
            nodata = opts.get("fill_value", _default_nodata(xx.dtype))
            xx = xx.groupby("time").map(stackstac.mosaic, nodata=nodata)

        if xx.odc.geobox.transform != xx.spec.transform:
            # work around issue 93 in stackstac
            xx.y.data[:] = xx.y.data - xx.spec.resolutions_xy[1]
    else:
        raise ValueError(f"Unsupported method:'{params.method}'")

    xx.attrs["load_params"] = params
    xx.attrs["_opts"] = opts
    return xx


def run_bench(
    xx: Union[xr.DataArray, xr.Dataset],
    client: distributed.Client,
    ntimes: int = 1,
    col_width: int = 12,
    restart_sleep: float = 0,
    results_file: Optional[str] = None,
) -> Tuple[BenchmarkContext, List[TimeSample]]:
    """
    Run same configuration multiple times and resport timing.

    :param xx: Dask graph to persist to ram
    :param client: Dask client to test on (will be restarted between runs)
    :param ntimes: How many rounds to run (default: 1)
    :param col_width: First column width in characters
    :param restart_sleep: Number of seconds to sleep after ``client.restart()``
    :param results_file: If set pickle results to this file, it will be overwritten
                         after every run.
    :returns: :class:`odc.stac.bench.BenchmarkContext` and timing info per run.

    Reported timing info is a triple of ``(t0, t_finished_submit, t_finished_persist)``
    """
    params = xx.attrs.get("load_params", None)

    extra = {}
    if params is not None:
        extra["method"] = params.method
        extra["scenario"] = params.scenario

    bench_ctx = collect_context_info(client, xx, **extra)
    samples = []
    _xx = None

    try:
        print(bench_ctx.render_txt(col_width))
        for _ in range(ntimes):
            client.restart()
            sleep(restart_sleep)

            t0 = t_now()

            _xx = client.persist(xx)
            t1 = t_now()

            _ = distributed.wait(_xx)
            t2 = t_now()

            times = (t0, t1, t2)
            print("-" * 60)
            print(bench_ctx.render_timing_info(times, col_width))
            samples.append(times)

            if results_file is not None:
                with open(results_file, "wb") as dst:
                    pickle.dump({"context": bench_ctx, "samples": samples}, dst)
    except KeyboardInterrupt:
        print("Aborting early upon request")
        if _xx is not None:
            client.cancel(_xx)

    return bench_ctx, samples


def _trim_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


def _method_from_string(method_path: str) -> Any:
    module_name, method_name = method_path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), method_name)
