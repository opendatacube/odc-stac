"""stac.load - dc.load from STAC Items."""

from __future__ import annotations

import dataclasses
import functools
import itertools
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeAlias,
    Union,
    cast,
)

import pystac
import pystac.item
import xarray as xr
from dask.utils import ndeepmap
from odc.geo import CRS, MaybeCRS, SomeResolution
from odc.geo.geobox import GeoBox, GeoboxAnchor, GeoboxTiles
from odc.geo.types import Unset
from odc.loader import chunked_load, resolve_chunk_shape, resolve_load_cfg
from odc.loader.types import Band_DType, ReaderDriverSpec

from ._mdtools import ConversionConfig, _resolve_driver, output_geobox, parse_items
from .model import BandQuery, ParsedItem, RasterCollectionMetadata

DEFAULT_CHUNK_FOR_LOAD = 2048
"""Used to partition load when not using Dask."""

GroupbyCallback: TypeAlias = Callable[[pystac.item.Item, ParsedItem, int], Any]

Groupby: TypeAlias = Union[str, GroupbyCallback]


def _collection(items: Iterable[ParsedItem]) -> RasterCollectionMetadata:
    for item in items:
        return item.collection
    raise ValueError("Can't load empty sequence")


def patch_urls(
    item: ParsedItem, edit: Callable[[str], str], bands: BandQuery = None
) -> ParsedItem:
    """
    Map function over dataset measurement urls.

    :param item: Item to edit in place
    :param edit: Function that returns modified url from input url
    :param bands: Only edit specified bands, default is to edit all
    :return: Input item
    """

    if bands is None:
        _bands = {
            k: dataclasses.replace(src, uri=edit(src.uri))
            for k, src in item.bands.items()
        }
    else:
        _to_edit = set(map(item.collection.band_key, bands))
        _bands = {
            k: dataclasses.replace(src, uri=edit(src.uri) if k in _to_edit else src.uri)
            for k, src in item.bands.items()
        }

    return dataclasses.replace(item, bands=_bands)


# pylint: disable=too-many-arguments,too-many-locals,too-many-statements
def load(
    items: Iterable[pystac.item.Item],
    bands: str | Sequence[str] | None = None,
    *,
    groupby: Groupby | None = "time",
    resampling: str | dict[str, str] | None = None,
    dtype: Band_DType = None,
    chunks: dict[str, int | Literal["auto"]] | None = None,
    pool: ThreadPoolExecutor | int | None = None,
    # Geo selection
    crs: MaybeCRS = Unset(),
    resolution: SomeResolution | None = None,
    anchor: GeoboxAnchor | None = None,
    geobox: GeoBox | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    lon: tuple[float, float] | None = None,
    lat: tuple[float, float] | None = None,
    x: tuple[float, float] | None = None,
    y: tuple[float, float] | None = None,
    like: Any = None,
    geopolygon: Any = None,
    intersects: Any = None,
    # UI
    progress: Any = None,
    fail_on_error: bool = True,
    # stac related
    stac_cfg: ConversionConfig | None = None,
    with_properties: Sequence[str | Mapping[str, Any]] | None = None,
    patch_url: Callable[[str], str] | None = None,
    preserve_original_order: bool = False,
    # custom driver
    driver: ReaderDriverSpec | None = None,
    # load behaviour
    fuse_func: str | Mapping[str, str | None] | None = None,
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
           query.items(),
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
       Controls what items get placed in to the same pixel plane.

       Following have special meaning:

       * "time" items with exactly the same timestamp are grouped together
       * "solar_day" items captured on the same day adjusted for solar time
       * "id" every item is loaded separately

       Any other string is assumed to be a key in Item's properties dictionary.

       You can also supply custom key function, it should take 3 arguments
       ``(pystac.Item, ParsedItem, index:int) -> Any``

    :param preserve_original_order:
       By default items are sorted by `time, id` within each group to make pixel
       fusing order deterministic. Setting this flag to ``True`` will instead keep
       items within each group in the same order as supplied, so that one can implement
       arbitrary priority for pixel overlap cases.

    :param resampling:
       Controls resampling strategy, can be specified per band

    :param dtype:
       Force output dtype, can be specified per band

    :param chunks:
       Rather than loading pixel data directly, construct
       Dask backed arrays. ``chunks={'x': 2048, 'y': 2048}``

    :param progress:
       Pass in ``tqdm`` progress bar or similar, only used in non-Dask load.

    :param fail_on_error:
        Set this to ``False`` to skip over load failures.

    :param pool:
       Use thread pool to perform load locally, only used in non-Dask load.

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
       Load data in a given CRS. Special name of ``"utm"`` is also understood, in which case an
       appropriate UTM projection will be picked based on the output bounding box.

    :param resolution:
       Set resolution of output in units of the output CRS. This can be a single float, in which
       case pixels are assumed to be square with ``Y`` axis flipped. To specify non-square or
       non-flipped pixels use :py:func:`odc.geo.resxy_` or :py:func:`odc.geo.resyx_`.
       ``resolution=10`` is equivalent to ``resolution=odc.geo.resxy_(10, -10)``.

       Resolution must be supplied in the units of the output CRS. Units are commonly meters
       for *Projected* and degrees for *Geographic* CRSs.

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

    :param anchor:
       Controls pixel snapping, default is to align pixel grid to ``X``/``Y``
       axis such that pixel edges align with ``x=0, y=0``. Other common option is to
       align pixel centers to ``0,0`` rather than edges.

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

    :param intersects:
       Simple alias to `geopolygon` so that the same inputs work for `pystac_client.Client.search`
       as they do here.

    .. rubric:: STAC Related Options

    :param stac_cfg: Controls interpretation of :py:class:`pystac.Item`. Mostly used to specify "missing"
       metadata like pixel data types.

    :param with_properties:
       List of properties to load from STAC item. Can be a list of strings or dictionaries with
       the following fields: ``.key``, ``.name``, ``.dtype``, ``.nodata``, ``.units``, ``.fuser``.

    :param patch_url:
       Optionally transform url of every band before loading

    .. rubric:: Load behaviour options

    :param driver:
       Optional. If provided, use the specified driver to load the data.

    :param fuse_func:
        Function used to fuse/combine/reduce data with the ``group_by`` parameter.

        By default, pixels are only copied where valid (i.e. not nodata) pixels
        have not yet been copied from previous items.

        If data (especially categorical data) appears wrong or unexpected in areas
        where items overlap, then an appropriate fuse_func may help.

        The fuse_func can perform specific combining steps and can be specified per band.

    .. rubric:: Custom fuser functions

    Custom fuse functions should be defined as follows:

    .. code-block:: python

            def my_fuser(dst: np.ndarray, src: np.ndarray) -> None:
                # Create a boolean mask array of pixels from this src array to copy.
                mask = pixels_to_copy(src)

                # Efficiently copy only masked pixels to dst.
                np.copyto(dst, src, where=mask)

    For an example of a more sophisticated fuser function, see
    https://github.com/GeoscienceAustralia/dea-notebooks/blob/77e9e3a05c104f4a0de91857905acce5853975b6/Tools/dea_tools/datahandling.py#L713

    Fuser functions are passed to odc-stac as importable strings (fully qualified Python
    names of top-level functions) so that they can be serialised to dask workers.

    In the following example, the ``my_fuser`` function is used for ``band0``, the default nodata-only
    fuser is used for ``band1`` and the ``other_fuser`` function is used for all other raster bands:

    .. code-block:: python

        data = odc.stac.load(...,
            fuse_func={
                "band0": "mymodule.my_fuser",
                "band1": None,
                "*": "mymodule.other_fuser",
            }
        )

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
           query.items(),
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

       some-other-collection:
         assets:
         #...

       "*": # Applies to all collections if not defined on a collection
         warnings: ignore  # ignore|all (default all)

    """
    # pylint: disable=unused-argument,too-many-branches
    if bands is None:
        # dc.load name for bands is measurements
        bands = kw.pop("measurements", None)

    # normalize args
    # dc.load compatible name for crs is `output_crs`
    if isinstance(crs, Unset) or crs is None:
        crs = cast(MaybeCRS, kw.pop("output_crs", None))

    if groupby is None:
        groupby = "id"

    rdr, md_parser = _resolve_driver(driver, stac_cfg, with_properties=with_properties)

    items = list(items)
    _parsed = list(parse_items(items, md_plugin=md_parser))

    # Check we have all the bands of interest
    # will raise ValueError if no such band/alias
    collection = _collection(_parsed)
    bands_to_load = collection.resolve_bands(bands)
    bands = list(bands_to_load)

    load_cfg = resolve_load_cfg(
        bands_to_load,
        resampling,
        dtype=dtype,
        use_overviews=kw.get("use_overviews", True),
        nodata=kw.get("nodata", None),
        fail_on_error=fail_on_error,
        fuse_func=fuse_func,
    )
    if patch_url is not None:
        _parsed = [patch_urls(item, edit=patch_url, bands=bands) for item in _parsed]

    if geopolygon is None and intersects is not None:
        geopolygon = intersects

    gbox = output_geobox(
        _parsed,
        bands=bands,
        crs=crs,
        resolution=resolution,
        anchor=anchor,
        align=kw.get("align", None),
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
        # TODO: handle no raster bands case here by creating some fake
        # geobox when only aux bands are present/requested for loading
        raise ValueError("Failed to auto-guess CRS/resolution.")

    # Time dimension
    ((mid_lon, _),) = gbox.extent.centroid.to_crs("epsg:4326").points
    _grouped_idx = _group_items(
        items,
        _parsed,
        groupby,
        mid_lon,
        preserve_original_order=preserve_original_order,
    )

    tss = _extract_timestamps(ndeepmap(2, lambda idx: _parsed[idx], _grouped_idx))
    meta = collection.meta_for(bands)

    if chunks is not None:
        chunk_shape = resolve_chunk_shape(
            len(tss),
            gbox,
            chunks,
            cfg=load_cfg,
            extra_dims=meta.extra_dims_full(),
        )
    else:
        chunk_shape = (1, DEFAULT_CHUNK_FOR_LOAD, DEFAULT_CHUNK_FOR_LOAD)

    # Spatio-temporal binning
    assert isinstance(gbox.crs, CRS)
    gbt = GeoboxTiles(gbox, (chunk_shape[1], chunk_shape[2]))
    tyx_bins = dict(_tyx_bins(_grouped_idx, _parsed, gbt))
    srcs = [item.resolve_bands(bands) for item in _parsed]
    debug = kw.get("debug", False)

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
                srcs=srcs,
                grouped_idx=_grouped_idx,
                tyx_bins=tyx_bins,
                bands_to_load=bands_to_load,
                load_cfg=load_cfg,
                **kw,
            )
        )
        return ds

    rdr_env = rdr.capture_env()
    return _with_debug_info(
        chunked_load(
            load_cfg,
            meta,
            srcs,
            tyx_bins,
            gbt,
            tss,
            rdr_env,
            rdr,
            chunks=chunks,
            pool=pool,
            progress=progress,
            dtype=dtype,
        )
    )


def _extract_timestamps(grouped: List[List[ParsedItem]]) -> List[datetime]:
    def _ts(group: List[ParsedItem]) -> datetime:
        assert len(group) > 0
        return group[0].nominal_datetime.replace(tzinfo=None)

    return list(map(_ts, grouped))


# pylint: disable=unused-argument
def _groupby_solar_day(
    item: pystac.item.Item,
    parsed: ParsedItem,
    idx: int,
    lon: Optional[float] = None,
):
    if lon is None:
        return parsed.solar_date.date()
    return parsed.solar_date_at(lon).date()


def _groupby_time(
    item: pystac.item.Item,
    parsed: ParsedItem,
    idx: int,
):
    return parsed.nominal_datetime


def _groupby_id(
    item: pystac.item.Item,
    parsed: ParsedItem,
    idx: int,
):
    return idx


def _groupby_property(
    item: pystac.item.Item,
    parsed: ParsedItem,
    idx: int,
    key: str = "",
):
    return item.properties.get(key, None)


def _resolve_groupby(groupby: Groupby, lon: Optional[float] = None) -> GroupbyCallback:
    if not isinstance(groupby, str):
        return groupby
    if groupby == "time":
        return _groupby_time
    if groupby == "solar_day":
        return functools.partial(_groupby_solar_day, lon=lon)
    if groupby == "id":
        return _groupby_id

    return functools.partial(_groupby_property, key=groupby)


def _group_items(
    items: List[pystac.item.Item],
    parsed: List[ParsedItem],
    groupby: Groupby,
    lon: Optional[float] = None,
    preserve_original_order=False,
) -> List[List[int]]:
    assert len(items) == len(parsed)

    group_key = _resolve_groupby(groupby, lon=lon)

    def _sorter(idx: int):
        _group = group_key(items[idx], parsed[idx], idx)

        if preserve_original_order:
            # Sort by group_key but keeping original item order within each group
            return (_group, idx)

        # Sort by group_key, but then time,id within each group
        return (_group, parsed[idx].nominal_datetime, parsed[idx].id)

    ii = sorted(range(len(parsed)), key=_sorter)

    return [
        list(group)
        for _, group in itertools.groupby(
            ii, lambda idx: group_key(items[idx], parsed[idx], idx)
        )
    ]


def _tiles(item: ParsedItem, gbt: GeoboxTiles) -> Iterator[Tuple[int, int]]:
    geom = item.safe_geometry(gbt.base.crs)
    if geom is None:
        raise ValueError("Can not process items without defined footprint")
    yield from gbt.tiles(geom)


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
