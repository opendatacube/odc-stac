"""stac.load - dc.load from STAC Items."""
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Set, Tuple, Union

import datacube.model
import datacube.utils.geometry
import numpy as np
import pyproj
import pystac
import pystac.item
import xarray
from affine import Affine
from datacube.model import Dataset
from datacube.storage import measurement_paths
from pyproj.crs.crs import CRS
from toolz import dicttoolz

from ._dcload import dc_load
from ._eo3 import ConversionConfig, stac2ds

SomeCRS = Union[str, datacube.utils.geometry.CRS, pyproj.CRS, Dict[str, Any]]
MaybeCRS = Optional[SomeCRS]


def eo3_geoboxes(
    ds: Dataset,
    bands: Optional[Sequence[str]] = None,
    grids: Optional[Sequence[str]] = None,
) -> Dict[str, datacube.utils.geometry.GeoBox]:
    """
    Extract EO3 grids in GeoBox format.

    :param dataset: EO3 Dataset
    :param bands: Optional list of bands of interest
    :param grids: Optional list of grids of interest

    :returns: a dictionary mapping grid names to a corresponding
              :class:`~datacube.utils.geometry.GeoBox`
    """
    crs = ds.crs
    _grids = ds.metadata_doc.get("grids", None)

    if _grids is None:
        raise ValueError("Missing grids, is this EO3 style Dataset?")
    if bands is not None:
        relevant_grids: Set[str] = set()
        for band in bands:
            band = ds.type.canonical_measurement(band)
            relevant_grids.add(ds.measurements[band].get("grid", "default"))
        grids = list(relevant_grids)

    if grids is not None:
        _grids = dicttoolz.keyfilter(lambda k: k in grids, _grids)

    def to_geobox(grid: Dict[str, Any]) -> datacube.utils.geometry.GeoBox:
        shape = grid.get("shape")
        transform = grid.get("transform")
        if shape is None or transform is None:
            raise ValueError("Each grid must have .shape and .transform")
        if len(shape) != 2:
            raise ValueError("Shape must contain `(height, width)`")
        if len(transform) not in (6, 9):
            raise ValueError(
                "Invalid `transform` specified, expect 6 or 9 element array"
            )
        h, w = shape
        return datacube.utils.geometry.GeoBox(w, h, Affine(*transform[:6]), crs)

    return dicttoolz.valmap(to_geobox, _grids)


def most_common_crs(crss: Iterable[CRS]) -> CRS:
    """
    Find most frequently occuring CRS.

    :param crss: Iterable of :class:`~datacube.utils.geometry.CRS` objects
    """
    _cc: Dict[CRS, int] = {}
    for crs in crss:
        _cc.setdefault(crs, 0)
        _cc[crs] += 1

    assert len(_cc) > 0

    # get CRS with highest count
    crs, _ = sorted(_cc.items(), reverse=True, key=(lambda kv: kv[1]))[0]
    return crs


def pick_best_resolution(
    dss: Sequence[Dataset], bands: Optional[Sequence[str]] = None
) -> Optional[Tuple[float, float]]:
    """
    Pick "best" resolution to use for data load.

    Given a non-empty sequence of :class:`~datacube.model.Dataset` objects and a
    set of bands to be loaded figure out what resolution is most appropriate.

    :param dss: Sequence of Dataset objects
    :param bands: Set of bands of interest, default: consider all bands.
    :return: ``(Y, X)`` resolution tuple
    """

    def best(
        a: Tuple[float, float], b: Optional[Tuple[float, float]]
    ) -> Tuple[float, float]:
        if b is None:
            return a
        a_min: float = min(map(abs, a))  # type: ignore
        b_min: float = min(map(abs, b))  # type: ignore

        return a if a_min <= b_min else b

    res_best = None

    for ds in dss:
        for geobox in eo3_geoboxes(ds, bands=bands).values():
            if geobox.shape != (1, 1):
                res_best = best(geobox.resolution, res_best)

    return res_best


def patch_urls(
    ds: Dataset, edit: Callable[[str], str], bands: Optional[Iterable[str]] = None
) -> Dataset:
    """
    Map function over dataset measurement urls.

    :param ds: Dataset to edit in place
    :param edit: Function that returns modified url from input url
    :param bands: Only edit specified bands, default is to edit all
    :return: Input dataset
    """
    resolved_paths = measurement_paths(ds)
    if bands is None:
        bands = list(resolved_paths)
    else:
        # remap aliases if present to their canonical name
        bands = list(map(ds.type.canonical_measurement, bands))

    mm = ds.metadata_doc["measurements"]
    for band in bands:
        mm[band]["path"] = edit(resolved_paths[band])
    return ds


# pylint: disable=too-many-arguments,too-many-locals
def load(
    items: Iterable[pystac.item.Item],
    bands: Optional[Union[str, Sequence[str]]] = None,
    *,
    groupby: Optional[str] = None,
    resampling: Optional[Union[str, Dict[str, str]]] = None,
    chunks: Optional[Dict[str, int]] = None,
    # Geo selection
    crs: MaybeCRS = None,
    resolution: Optional[Union[float, int, Tuple[float, float]]] = None,
    geobox: Optional[datacube.utils.geometry.GeoBox] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    lon: Optional[Tuple[float, float]] = None,
    lat: Optional[Tuple[float, float]] = None,
    x: Optional[Tuple[float, float]] = None,
    y: Optional[Tuple[float, float]] = None,
    align: Optional[Union[float, int, Tuple[float, float]]] = None,
    like: Optional[Any] = None,
    geopolygon: Optional[Any] = None,
    # stac related
    stac_cfg: Optional[ConversionConfig] = None,
    patch_url: Optional[Callable[[str], str]] = None,
    product_cache: Optional[Dict[str, datacube.model.DatasetType]] = None,
    # dc.load pass-through args
    skip_broken_datasets: bool = False,
    progress_cbk: Optional[Callable[[int, int], Any]] = None,
    fuse_func: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    **kw,
) -> xarray.Dataset:
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
           crs="EPSG:32606",
           resolution=(-100, 100),
       )
       xx.red.plot.imshow(col="time")


    :param items:
       Iterable of STAC :class:`~pystac.Item` to load

    :param bands:
       List of band names to load, defaults to All. Also accepts
       single band name as input

    .. rubric:: Common Options

    :param groupby:
       Controls what items get placed in to the same pixel plane,
       supported values are "time" or "solar_day", default is "time"

    :param resampling:
       Controls resampling strategy, can be specified per band

    :param chunks:
       Rather than loading pixel data directly, construct
       Dask backed arrays. ``chunks={'x': 2048, 'y': 2048}``

    .. rubric:: Control Pixel Grid of Output

    There are many ways to control footprint and resolution of returned data. The most
    precise way is to use :py:class:`~datacube.utils.geometry.GeoBox`, ``geobox=GeoBox(..)``.
    Similarly one can use ``like=xx`` to match pixel grid to previously loaded data
    (``xx = odc.stac.load(...)``).

    Other common way is to configure crs and resolution only

    .. code-block:: python

       xx = odc.stac.load(...
           crs="EPSG:3857",
           resolution=(-10, 10))

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
       :class:`~datacube.utils.geometry.GeoBox` object

    :param like:
       Match output grid to the data loaded previously.

    :param geopolygon:
       Limit returned result to a bounding box of a given geometry. This could be an
       instance of :class:`~datacube.utils.geometry.Geometry`, GeoJSON dictionary,
       GeoPandas DataFrame, or any object implementing ``__geo_interface__``. We assume
       ``EPSG:4326`` projection for dictionary and Shapely inputs. CRS information available
       on GeoPandas inputs should be understood correctly.

    .. rubric:: STAC Related Options

    :param stac_cfg:
       Controls :class:`pystac.Item` ``->`` :class:`datacube.model.Dataset`
       conversion, mostly used to specify "missing" metadata like pixel
       data types.

       See :func:`~odc.stac.stac2ds` and section below for more details.

    :param patch_url:
       Optionally transform url of every band before loading

    :param product_cache:
       Passed on to :func:`~odc.stac.stac2ds`

    .. rubric:: Pass-through to :py:meth:`datacube.Datacube.load`

    :param progress_cbk:
       Get data loading progress via callback, ignored when
       constructing Dask arrays

    :param skip_broken_datasets:
       Continue processing when IO errors are encountered

    :param fuse_func:
       Provide custom function for fusing pixels from different
       sources into one pixel plane.

       The default behaviour is to use first observed valid pixel.
       Item timestamp is used to determine order, ``nodata`` is
       used to determine "valid".

    :param kw:
       Any other named parameter is passed on to :py:meth:`datacube.Datacube.load`

    :return:
       :class:`xarray.Dataset` with requested bands populated


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
           crs="EPSG:32606",
           resolution=100,
           patch_url=pc.sign,
       )
       xx.red.plot.imshow(col="time", size=8, aspect=1)


    .. rubric:: Example Configuration

    Sample ``stac_cfg={..}`` parameter.

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


    .. seealso::

       | STAC item interpretation :func:`~odc.stac.stac2ds`
       | Data loading: :py:meth:`datacube.Datacube.load`
    """
    if bands is None:
        # dc.load name for bands is measurements
        bands = kw.pop("measurements", None)

    if isinstance(resolution, (float, int)):
        resolution = (-float(resolution), float(resolution))

    if isinstance(align, (float, int)):
        align = (align, align)

    # STAC compatibility
    if bbox is not None:
        if any(v is not None for v in [x, y, lon, lat]):
            raise ValueError(
                "When supplying `bbox` you should not supply `x,y` or `lon,lat`"
            )
        x1, y1, x2, y2 = bbox
        lon = (x1, x2)
        lat = (y1, y2)

    # normalize args
    # dc.load has distinction between query crs and output_crs
    # but output_crs name can be confusing, especially that resolution is not output_resolution,
    # so we treat crs same as output_crs
    output_crs: MaybeCRS = kw.pop("output_crs", None)
    if output_crs is None and crs is not None:
        output_crs, crs = crs, None

    geo = dicttoolz.valfilter(
        lambda x: x is not None,
        dict(
            x=x,
            y=y,
            lon=lon,
            lat=lat,
            crs=crs,
            output_crs=output_crs,
            resolution=resolution,
            align=align,
            like=like,
            geopolygon=geopolygon,
            geobox=geobox,
        ),
    )

    def auto_fill_geo(geo, dss, bands):
        if "geobox" in geo:
            return
        if "like" in geo:
            return

        if "output_crs" not in geo:
            # Need to pick CRS
            geo["output_crs"] = most_common_crs(ds.crs for ds in dss)

        if "resolution" not in geo:
            # Need to pick resolution
            geo["resolution"] = pick_best_resolution(dss, bands)

    dss = list(stac2ds(items, stac_cfg, product_cache=product_cache))
    auto_fill_geo(geo, dss, bands)

    if patch_url is not None:
        dss = [patch_urls(ds, edit=patch_url, bands=bands) for ds in dss]

    return dc_load(
        dss,
        measurements=bands,
        groupby=groupby,
        resampling=resampling,
        chunks=chunks,
        progress_cbk=progress_cbk,
        skip_broken_datasets=skip_broken_datasets,
        fuse_func=fuse_func,
        **geo,
        **kw,
    )
