"""stac.load - dc.load from STAC Items."""
import dataclasses
import itertools
from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    Iterator,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    cast,
)

import numpy as np
import pystac
import pystac.item
import xarray as xr
from dask import array as da
from dask.base import quote, tokenize
from dask.utils import ndeepmap
from odc.geo import CRS, XY, MaybeCRS, SomeResolution
from odc.geo.geobox import GeoBox, GeoboxTiles
from odc.geo.xr import xr_coords
from xarray.core.npcompat import DTypeLike

from ._dask import unpack_chunks
from ._mdtools import ConversionConfig, output_geobox, parse_items, with_default
from ._model import (
    ParsedItem,
    RasterBandMetadata,
    RasterCollectionMetadata,
    RasterLoadParams,
    RasterSource,
)
from ._reader import _nodata_mask, _resolve_src_nodata, rio_read
from ._rio import _CFG, GDAL_CLOUD_DEFAULTS, get_rio_env, rio_env

DEFAULT_CHUNK_FOR_LOAD = 2048
"""Used to partition load when not using Dask."""


class MkArray(Protocol):
    """Internal interface."""

    # pylint: disable=too-few-public-methods
    def __call__(
        self,
        shape: Tuple[int, ...],
        dtype: DTypeLike,
        /,
        name: Hashable,
    ) -> Any:
        ...  # pragma: no cover


@dataclasses.dataclass(frozen=True)
class _LoadChunkTask:
    band: str
    srcs: List[Tuple[int, str]]
    cfg: RasterLoadParams
    gbt: GeoboxTiles
    idx_tyx: Tuple[int, int, int]

    @property
    def dst_roi(self):
        t, y, x = self.idx_tyx
        return (t, *self.gbt.roi[y, x])

    @property
    def dst_gbox(self) -> GeoBox:
        _, y, x = self.idx_tyx
        return self.gbt[y, x]


class _DaskGraphBuilder:
    # pylint: disable=too-few-public-methods

    def __init__(
        self,
        cfg: Dict[str, RasterLoadParams],
        items: List[ParsedItem],
        tyx_bins: Dict[Tuple[int, int, int], List[int]],
        gbt: GeoboxTiles,
        env: Dict[str, Any],
    ) -> None:
        self.cfg = cfg
        self.items = items
        self.tyx_bins = tyx_bins
        self.gbt = gbt
        self.env = env
        self._tk = tokenize(items, cfg, gbt, tyx_bins, env)

    def __call__(
        self,
        shape: Tuple[int, ...],
        dtype: DTypeLike,
        /,
        name: Hashable,
    ) -> Any:
        # pylint: disable=too-many-locals
        assert len(shape) == 3
        assert isinstance(name, str)
        cfg = self.cfg[name]
        assert dtype == cfg.dtype

        cfg_key = f"cfg-{tokenize(cfg)}"
        gbt_key = f"grid-{tokenize(self.gbt)}"

        dsk: Dict[Hashable, Any] = {
            cfg_key: cfg,
            gbt_key: self.gbt,
        }
        tk = self._tk
        band_key = f"{name}-{tk}"
        md_key = f"md-{name}-{tk}"
        shape_in_blocks = (shape[0], *self.gbt.shape.yx)
        for idx, item in enumerate(self.items):
            dsk[md_key, idx] = item[name]

        for ti, yi, xi in np.ndindex(shape_in_blocks):
            tyx_idx = (ti, yi, xi)
            srcs = [(md_key, idx) for idx in self.tyx_bins.get(tyx_idx, [])]
            dsk[band_key, ti, yi, xi] = (
                _dask_loader_tyx,
                srcs,
                gbt_key,
                quote((yi, xi)),
                cfg_key,
                self.env,
            )

        chunk_shape = (1, *self.gbt.chunk_shape((0, 0)).yx)
        chunks = unpack_chunks(chunk_shape, shape)

        return da.Array(dsk, band_key, chunks, dtype=dtype, shape=shape)


def _collection(items: Iterable[ParsedItem]) -> RasterCollectionMetadata:
    for item in items:
        return item.collection
    raise ValueError("Can't load empty sequence")


def patch_urls(
    item: ParsedItem, edit: Callable[[str], str], bands: Optional[Iterable[str]] = None
) -> ParsedItem:
    """
    Map function over dataset measurement urls.

    :param ds: Dataset to edit in place
    :param edit: Function that returns modified url from input url
    :param bands: Only edit specified bands, default is to edit all
    :return: Input dataset
    """

    if bands is None:
        _bands = {
            k: dataclasses.replace(src, uri=edit(src.uri))
            for k, src in item.bands.items()
        }
    else:
        aliases = item.collection.aliases
        bands = set(aliases.get(b, b) for b in bands)
        _bands = {
            k: dataclasses.replace(src, uri=edit(src.uri) if k in bands else src.uri)
            for k, src in item.bands.items()
        }

    return dataclasses.replace(item, bands=_bands)


def _capture_rio_env() -> Dict[str, Any]:
    # pylint: disable=protected-access
    if _CFG._configured:
        env = {**_CFG._gdal_opts, "_aws": _CFG._aws}
    else:
        env = get_rio_env(sanitize=False, no_session_keys=True)

    if len(env) == 0:
        # not customized, supply defaults
        return {**GDAL_CLOUD_DEFAULTS}

    # don't want that copied across to workers who might be on different machine
    env.pop("GDAL_DATA", None)
    return env


# pylint: disable=too-many-arguments,too-many-locals,too-many-statements
def load(
    items: Iterable[pystac.item.Item],
    bands: Optional[Union[str, Sequence[str]]] = None,
    *,
    groupby: Optional[str] = "time",
    resampling: Optional[Union[str, Dict[str, str]]] = None,
    dtype: Union[DTypeLike, Dict[str, DTypeLike], None] = None,
    chunks: Optional[Dict[str, int]] = None,
    # Geo selection
    crs: MaybeCRS = None,
    resolution: Optional[SomeResolution] = None,
    align: Optional[Union[float, int, XY[float]]] = None,
    geobox: Optional[GeoBox] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    lon: Optional[Tuple[float, float]] = None,
    lat: Optional[Tuple[float, float]] = None,
    x: Optional[Tuple[float, float]] = None,
    y: Optional[Tuple[float, float]] = None,
    like: Optional[Any] = None,
    geopolygon: Optional[Any] = None,
    # stac related
    stac_cfg: Optional[ConversionConfig] = None,
    patch_url: Optional[Callable[[str], str]] = None,
    **kw,
) -> xr.Dataset:
    """
    STAC :class:`~pystac.item.Item` to :class:`xarray.Dataset`.

    Load several STAC :class:`~pystac.item.Item` objects (from the same or similar
    collections) as an :class:`xarray.Dataset`.

    This method can load pixel data directly on a local machine or construct a Dask
    graph that can be processed on a remote cluster.

    .. code-block:: python

       catalog = pystac.Client.open(...)
       query = catalog.search(...)
       xx = odc.stac.load(
           query.get_items(),
           bands=["red", "green", "blue"],
       )
       xx.red.plot.imshow(col="time")


    :param items:
       Iterable of STAC :class:`~pystac.item.Item` to load

    :param bands:
       List of band names to load, defaults to All. Also accepts
       single band name as input

    .. rubric:: Common Options

    :param groupby:
       Controls what items get placed in to the same pixel plane,
       supported values are "time", "solar_day" and "id",
       default is "time"

    :param resampling:
       Controls resampling strategy, can be specified per band

    :param dtype:
       Force output dtype, can be specified per band

    :param chunks:
       Rather than loading pixel data directly, construct
       Dask backed arrays. ``chunks={'x': 2048, 'y': 2048}``

    .. rubric:: Control Pixel Grid of Output

    There are many ways to control footprint and resolution of returned data. The most
    precise way is to use :py:class:`~odc.geo.geobox.GeoBox`, ``geobox=GeoBox(..)``.
    Similarly one can use ``like=xx`` to match pixel grid to previously loaded data
    (``xx = odc.stac.load(...)``).

    Other common way is to configure crs and resolution only

    .. code-block:: python

       xx = odc.stac.load(...
           crs="EPSG:3857",
           resolution=10)

       # resolution units must match CRS
       # here we assume 1 degree == 111km to load at roughly
       # the same 10m resolution as statement above.
       yy = odc.stac.load(...
           crs="EPSG:4326",
           resolution=0.00009009)

    By default :py:func:`odc.stac.load` loads all available pixels in the requested
    projection and resolution. To limit extent of loaded data you have to supply bounds via
    either ``geobox=`` or ``like=`` parameters (these also select projection and resolution).
    Alternatively use a pair of ``x, y`` or ``lon, lat`` parameters. ``x, y`` allows you to
    specify bounds in the output projection, while ``lon, lat`` operate in degrees. You can also
    use ``bbox`` which is equivalent to ``lon, lat``.

    It should be noted that returned data is likely to reach outside of the requested bounds by
    fraction of a pixel when using ``bbox``, ``x, y`` or ``lon, lat`` mechanisms. This is due to
    pixel grid "snapping". Pixel edges will still start at ``N*pixel_size`` where ``N is int``
    regardless of the requested bounding box.

    :param crs:
       Load data in a given CRS

    :param resolution:
       Set resolution of output in ``Y, X`` order, it is common for ``Y`` to be negative,
       e.g. ``resolution=(-10, 10)``. Resolution must be supplied in the units of the
       output CRS, so they are commonly in meters for *Projected* and in degrees for
       *Geographic* CRSs. ``resolution=10`` is equivalent to ``resolution=(-10, 10)``.

    :param bbox:
       Specify bounding box in Lon/Lat. ``[min(lon), min(lat), max(lon), max(lat)]``

    :param lon:
       Define output bounds in Lon/Lat
    :param lat:
       Define output bounds in Lon/Lat

    :param x:
       Define output bounds in output projection coordinate units
    :param y:
       Define output bounds in output projection coordinate units

    :param align:
       Control pixel snapping, default is to align pixel grid to ``X``/``Y``
       axis such that pixel edges lie on the axis.

    :param geobox:
       Allows to specify exact region/resolution/projection using
       :class:`~odc.geo.geobox.GeoBox` object

    :param like:
       Match output grid to the data loaded previously.

    :param geopolygon:
       Limit returned result to a bounding box of a given geometry. This could be an
       instance of :class:`~odc.geo.geom.Geometry`, GeoJSON dictionary,
       GeoPandas DataFrame, or any object implementing ``__geo_interface__``. We assume
       ``EPSG:4326`` projection for dictionary and Shapely inputs. CRS information available
       on GeoPandas inputs should be understood correctly.

    .. rubric:: STAC Related Options

    :param stac_cfg:
       Controls interpretation of :py:class:`pystac.Item`. Mostly used to specify "missing"
       metadata like pixel data types.

    :param patch_url:
       Optionally transform url of every band before loading

    :return:
       :py:class:`xarray.Dataset` with requested bands populated


    .. rubric:: Complete Example Code

    .. code-block:: python

       import planetary_computer as pc
       from pystac_client import Client

       from odc import stac

       catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
       query = catalog.search(
           collections=["sentinel-2-l2a"],
           datetime="2019-06-06",
           query={"s2:mgrs_tile": dict(eq="06VVN")},
       )

       xx = stac.load(
           query.get_items(),
           bands=["red", "green", "blue"],
           resolution=100,  # 1/10 of the native 10m resolution
           patch_url=pc.sign,
       )
       xx.red.plot.imshow(col="time", size=8, aspect=1)


    .. rubric:: Example Optional Configuration

    Sometimes data source might be missing some optional STAC extensions. With ``stac_cfg=`` parameter
    one can supply that information at load time. Configuration is per collection per asset. You can
    provide information like pixel data type, ``nodata`` value used, ``unit`` attribute and band aliases
    you would like to use.

    Sample ``stac_cfg={..}`` parameter:

    .. code-block:: yaml

       sentinel-2-l2a:  # < name of the collection, i.e. ``.collection_id``
         assets:
           "*":  # Band named "*" contains band info for "most" bands
             data_type: uint16
             nodata: 0
             unit: "1"
           SCL:  # Those bands that are different than "most"
             data_type: uint8
             nodata: 0
             unit: "1"
         aliases:  #< unique alias -> canonical map
           rededge: B05
           rededge1: B05
           rededge2: B06
           rededge3: B07

         warnings: ignore  # ignore|all  (default all)

       some-other-collection:
         assets:
         #...

       "*": # Applies to all collections if not defined on a collection
         warnings: ignore

    """
    # pylint: disable=unused-argument,too-many-branches
    if bands is None:
        # dc.load name for bands is measurements
        bands = kw.pop("measurements", None)

    # normalize args
    # dc.load compatible name for crs is `output_crs`
    if crs is None:
        crs = cast(MaybeCRS, kw.pop("output_crs", None))

    if groupby is None:
        groupby = "id"

    _parsed = list(parse_items(items, cfg=stac_cfg))

    gbox = output_geobox(
        _parsed,
        bands=bands,
        crs=crs,
        resolution=resolution,
        align=align,
        geobox=geobox,
        like=like,
        geopolygon=geopolygon,
        bbox=bbox,
        lon=lon,
        lat=lat,
        x=x,
        y=y,
    )

    if gbox is None:
        raise ValueError("Failed to auto-guess CRS/resolution.")

    if chunks is not None:
        chunk_shape = _resolve_chunk_shape(gbox, chunks)
    else:
        chunk_shape = _resolve_chunk_shape(
            gbox,
            {dim: DEFAULT_CHUNK_FOR_LOAD for dim in gbox.dimensions},
        )

    debug = kw.get("debug", False)

    # Check we have all the bands of interest
    # will raise ValueError if no such band/alias
    collection = _collection(_parsed)
    bands_to_load = collection.resolve_bands(bands)
    bands = list(bands_to_load)

    load_cfg = _resolve_load_cfg(
        bands_to_load,
        resampling,
        dtype=dtype,
        use_overviews=kw.get("use_overviews", True),
        nodata=kw.get("nodata", None),
    )

    if patch_url is not None:
        _parsed = [patch_urls(item, edit=patch_url, bands=bands) for item in _parsed]

    # Time dimension
    ((mid_lon, _),) = gbox.extent.centroid.to_crs("epsg:4326").points
    _grouped_idx = _group_items(_parsed, groupby, mid_lon)

    tss = _extract_timestamps(ndeepmap(2, lambda idx: _parsed[idx], _grouped_idx))

    # Spatio-temporal binning
    assert isinstance(gbox.crs, CRS)
    gbt = GeoboxTiles(gbox, chunk_shape)
    tyx_bins = dict(_tyx_bins(_grouped_idx, _parsed, gbt))
    _parsed = [item.strip() for item in _parsed]

    def _with_debug_info(ds: xr.Dataset, **kw) -> xr.Dataset:
        # expose data for debugging
        if not debug:
            return ds

        from types import SimpleNamespace  # pylint: disable=import-outside-toplevel

        ds.encoding.update(
            debug=SimpleNamespace(
                gbt=gbt,
                mid_lon=mid_lon,
                parsed=_parsed,
                grouped_idx=_grouped_idx,
                tyx_bins=tyx_bins,
                bands_to_load=bands_to_load,
                load_cfg=load_cfg,
                **kw,
            )
        )
        return ds

    def _task_stream(bands: List[str]) -> Iterator[_LoadChunkTask]:
        _shape = (len(_grouped_idx), *gbt.shape)
        for band_name in bands:
            cfg = load_cfg[band_name]
            for ti, yi, xi in np.ndindex(_shape):
                tyx_idx = (ti, yi, xi)
                srcs = [(idx, band_name) for idx in tyx_bins.get(tyx_idx, [])]
                yield _LoadChunkTask(band_name, srcs, cfg, gbt, tyx_idx)

    if chunks is not None:
        # Dask case: dummy for now
        _loader = _DaskGraphBuilder(
            load_cfg,
            _parsed,
            tyx_bins,
            gbt,
            _capture_rio_env(),
        )
        return _with_debug_info(_mk_dataset(gbox, tss, load_cfg, _loader))

    ds = _mk_dataset(gbox, tss, load_cfg)
    _tasks = []

    for task in _task_stream(bands):
        if debug:
            print(f"{task.band}[{task.idx_tyx}]")
            _tasks.append(task)

        dst_slice = ds[task.band].data[task.dst_roi]
        srcs = [_parsed[idx][band] for idx, band in task.srcs]
        _ = _fill_2d_slice(srcs, task.dst_gbox, task.cfg, dst_slice)

    return _with_debug_info(ds, tasks=_tasks)


def _resolve_load_cfg(
    bands: Dict[str, RasterBandMetadata],
    resampling: Optional[Union[str, Dict[str, str]]] = None,
    dtype: Union[DTypeLike, Dict[str, DTypeLike], None] = None,
    use_overviews: bool = True,
    nodata: Optional[float] = None,
) -> Dict[str, RasterLoadParams]:
    def _dtype(name: str, band_dtype: Optional[str], fallback: str) -> str:
        if dtype is None:
            return with_default(band_dtype, fallback)
        if isinstance(dtype, dict):
            return str(
                with_default(
                    dtype.get(name, dtype.get("*", band_dtype)),
                    fallback,
                )
            )
        return str(dtype)

    def _resampling(name: str, fallback: str) -> str:
        if resampling is None:
            return fallback
        if isinstance(resampling, dict):
            return resampling.get(name, resampling.get("*", fallback))
        return resampling

    def _fill_value(band: RasterBandMetadata) -> Optional[float]:
        if nodata is not None:
            return nodata
        return band.nodata

    def _resolve(name: str, band: RasterBandMetadata) -> RasterLoadParams:
        return RasterLoadParams(
            _dtype(name, band.data_type, "float32"),
            fill_value=_fill_value(band),
            use_overviews=use_overviews,
            resampling=_resampling(name, "nearest"),
        )

    return {name: _resolve(name, band) for name, band in bands.items()}


def _dask_loader_tyx(
    srcs: List[RasterSource],
    gbt: GeoboxTiles,
    iyx: Tuple[int, int],
    cfg: RasterLoadParams,
    env: Dict[str, Any],
):
    assert cfg.dtype is not None
    env = {**env}
    session = env.pop("_aws", None)
    gbox = gbt[iyx]
    chunk = np.empty(gbox.shape.yx, dtype=cfg.dtype)
    with rio_env(session, **env):
        return _fill_2d_slice(srcs, gbox, cfg, chunk)[np.newaxis]


def _fill_2d_slice(
    srcs: List[RasterSource],
    dst_gbox: GeoBox,
    cfg: RasterLoadParams,
    dst: Any,
) -> Any:
    # TODO: support masks not just nodata based fusing
    #
    # ``nodata``     marks missing pixels, but it might be None (everything is valid)
    # ``fill_value`` is the initial value to use, it's equal to ``nodata`` when set,
    #                otherwise defaults to .nan for floats and 0 for integers
    assert dst.shape == dst_gbox.shape.yx
    nodata = _resolve_src_nodata(cfg.fill_value, cfg)

    if nodata is None:
        fill_value = float("nan") if dst.dtype.kind == "f" else 0
    else:
        fill_value = nodata

    np.copyto(dst, fill_value)
    if len(srcs) == 0:
        return dst

    src, *rest = srcs
    _roi, pix = rio_read(src, cfg, dst_gbox, dst=dst)

    for src in rest:
        # first valid pixel takes precedence over others
        _roi, pix = rio_read(src, cfg, dst_gbox)

        # nodata mask takes care of nan when working with floats
        # so you can still get proper mask even when nodata is None
        # when working with float32 data.
        missing = _nodata_mask(dst[_roi], nodata)
        np.copyto(dst[_roi], pix, where=missing)

    return dst


def _mk_dataset(
    gbox: GeoBox,
    time: List[datetime],
    bands: Dict[str, RasterLoadParams],
    alloc: Optional[MkArray] = None,
) -> xr.Dataset:
    _shape = (len(time), *gbox.shape.yx)
    coords = xr_coords(gbox)
    crs_coord_name: Hashable = list(coords)[-1]
    coords["time"] = xr.DataArray(time, dims=("time",))
    dims = ("time", *gbox.dimensions)

    def _alloc(shape: Tuple[int, ...], dtype: str, name: Hashable) -> Any:
        if alloc is not None:
            return alloc(shape, dtype, name=name)
        return np.empty(shape, dtype=dtype)

    def _maker(name: Hashable, band: RasterLoadParams) -> xr.DataArray:
        assert band.dtype is not None
        data = _alloc(_shape, band.dtype, name=name)
        attrs = {}
        if band.fill_value is not None:
            attrs["nodata"] = band.fill_value

        xx = xr.DataArray(data=data, coords=coords, dims=dims, attrs=attrs)
        xx.encoding.update(grid_mapping=crs_coord_name)
        return xx

    return xr.Dataset({name: _maker(name, band) for name, band in bands.items()})


def _extract_timestamps(grouped: List[List[ParsedItem]]) -> List[datetime]:
    def _ts(group: List[ParsedItem]) -> datetime:
        assert len(group) > 0
        return group[0].nominal_datetime.replace(tzinfo=None)

    return list(map(_ts, grouped))


def _group_items(
    items: List[ParsedItem],
    groupby: str,
    lon: Optional[float] = None,
) -> List[List[int]]:
    def _time(idx: int):
        # group by timestamp, sort by (timestamp, id)
        xx = items[idx]
        return (xx.nominal_datetime, xx.id)

    def _solar_day(idx: int):
        # group by solar day date component, but sort by (solar day timestamp, id)
        xx = items[idx]
        if lon is None:
            ts = xx.solar_date
        else:
            ts = xx.solar_date_at(lon)
        return (ts.date(), ts, xx.id)

    key = {
        "id": _time,
        "time": _time,
        "solar_day": _solar_day,
    }.get(groupby, None)

    if key is None:
        raise ValueError(f"Groupby '{groupby}' is not a valid option.")

    assert key is not None
    ii = sorted(range(len(items)), key=key)

    if groupby == "id":
        return [[idx] for idx in ii]

    grouper = lambda xx: key(xx)[0]

    return [list(group) for _, group in itertools.groupby(ii, grouper)]


def _tiles(item: ParsedItem, gbt: GeoboxTiles) -> Iterator[Tuple[int, int]]:
    # TODO: should probably prefer native geometry if set in proj
    # TODO: extract geometry from geobox if proj data is available
    if item.geometry is None:
        raise ValueError("Can not process items without defined footprint")
    yield from gbt.tiles(item.geometry)


def _tyx_bins(
    grouped: List[List[int]],
    items: List[ParsedItem],
    gbt: GeoboxTiles,
) -> Iterator[Tuple[Tuple[int, int, int], List[int]]]:
    for t_idx, group in enumerate(grouped):
        _yx: Dict[Tuple[int, int], List[int]] = {}

        for item_idx in group:
            for idx in _tiles(items[item_idx], gbt):
                _yx.setdefault(idx, []).append(item_idx)

        yield from (((t_idx, *idx), ii_item) for idx, ii_item in _yx.items())


def _resolve_chunk_shape(gbox: GeoBox, chunks: Dict[str, int]) -> Tuple[int, int]:
    def _norm_dim(chunk: int, sz: int) -> int:
        if chunk < 0 or chunk > sz:
            return sz
        return chunk

    ny, nx = [
        _norm_dim(chunks.get(dim, chunks.get(fallback_dim, -1)), n)
        for dim, fallback_dim, n in zip(gbox.dimensions, ["y", "x"], gbox.shape.yx)
    ]
    return ny, nx
