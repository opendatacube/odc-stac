"""Utilities for benchmarking."""
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, Union

import affine
import distributed
import numpy as np
import pystac.item
import xarray as xr
from dask.utils import format_bytes

import odc.stac

# pylint: disable=too-many-instance-attributes,too-many-locals,import-outside-toplevel,import-error


@dataclass
class BenchmarkContext:
    """
    Bencmark Context Metadata.

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

    def render_txt(self, col_width: int = 10) -> str:
        """
        Render textual representation for human consumption.

        :param col_width: Left column width in characters, defaults to 10
        :return: Multiline string representation of self
        """
        nw = col_width
        data_dims = ".".join(map(str, self.shape))
        chunk_dims = ".".join(map(str, self.chunks))

        transorm_txt = f"\n{'':{nw}}".join(str(self.transform).split("\n")[:2])
        transorm_txt = transorm_txt.replace(".00,", ",")
        transorm_txt = transorm_txt.replace(".00|", "|")

        return f"""
{"method":{nw}}: {self.method}
{"Scenario":{nw}}: {self.scenario}
{"T.slice":{nw}}: {self.temporal_id}
{"Data":{nw}}: {data_dims}.{self.dtype},  {format_bytes(self.nbytes)}
{"Chunks":{nw}}: {chunk_dims} (T.B.Y.X)
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
        _chunks = {k: max(v) for k, v in xx.chunksizes.items()}
    elif isinstance(xx, xr.Dataset):
        npix = sum(b.data.size for b in xx.data_vars.values())
        nbytes = sum(b.data.dtype.itemsize * b.data.size for b in xx.data_vars.values())
        dtype, *_ = {b.dtype for b in xx.data_vars.values()}
        nb = len(xx.data_vars)
        sample_band, *_ = xx.data_vars.values()
        _chunks = {k: max(v) for k, v in sample_band.chunksizes.items()}
    else:
        raise ValueError("Expect one of `xarray.{DataArray,Dataset}` on input")

    yx_dims = tuple(xx.dims)[-2:]
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
    geobox = getattr(xx, "geobox", None)
    if geobox is None:
        raise ValueError("Can't find GEO info")
    crs = f"epsg:{xx.geobox.crs.epsg}"

    transform = xx.geobox.transform

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


def load_from_json(geojson, method: str = "odc-stac", patch_url=None, **kw):
    """
    Turn passed in geojson into a Dask array.

    :param geojson: GeoJSON FeatureCollection
    :param method: odc-stac (default)| stackstac
    """
    all_items = [pystac.item.Item.from_dict(f) for f in geojson["features"]]

    if method == "odc-stac":
        opts = dict(**kw)
        opts.setdefault("chunks", {})  # force Dask

        xx = odc.stac.load(all_items, patch_url=patch_url, **opts)
    elif method == "stackstac":
        import stackstac

        if patch_url is None:
            patch_url = lambda x: x

        _items = [patch_url(item).to_dict() for item in all_items]
        opts = dict(**kw)
        opts.setdefault("dtype", "uint16")
        opts.setdefault("fill_value", 0)
        opts.setdefault("xy_coords", "center")
        xx = stackstac.stack(_items, **opts)
        if np.unique(xx.time.data).shape != xx.time.shape:
            xx = xx.groupby("time").map(stackstac.mosaic)

        if xx.geobox.transform != xx.spec.transform:
            # work around issue 93 in stackstac
            xx.y.data[:] = xx.y.data - xx.spec.resolutions_xy[1]
    else:
        raise ValueError(f"Unsupported method:'{method}'")

    return xx
