"""
stac.load - dc.load from STAC Items
"""
from functools import partial
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pyproj
import pystac
from toolz import dicttoolz
import xarray
import datacube.utils.geometry
from odc.index import patch_urls

from ._dcload import dc_load
from ._eo3 import ConversionConfig, stac2ds

SomeCRS = Union[str, datacube.utils.geometry.CRS, pyproj.CRS, Dict[str, Any]]
MaybeCRS = Optional[SomeCRS]


# pylint: disable=too-many-arguments,too-many-locals
def load(
    items: Iterable[pystac.Item],
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
    output_crs: MaybeCRS = None,
    like: Optional[Any] = None,
    geopolygon: Optional[datacube.utils.geometry.Geometry] = None,
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
    Load several STAC :class:`pystac.Item` objects (from the same or similar
    collections) as an :class:`xarray.Dataset`

    This method can load pixel data directly locally or construct Dask graph that can
    be processed on a remote cluster.

    .. code-block:: python

       catalog = pystac.Client.open(...)
       query = catalog.search(...)
       xx = odc.stac.load(
           query.get_items(),
           bands=["red", "green", "blue"],
           crs="EPSG:32606",
           resolution=(-100, 100),
       )
       xx.red.plot.imshow(col="time");


    .. note::

       At the moment one must specify desired projection and resolution of
       the result. The plan is to make this choice automatic if not configured.


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

    :param output_crs:
       Same as ``crs``, name used by the underlying
       :py:meth:`~datacube.Datacube.load` method.

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
       from odc import stac
       from pystac_client import Client

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
       xx.red.plot.imshow(col="time", size=8, aspect=1);


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
         uuid:   # Rules for constructing UUID for Datasets (PLANNED, not implemented yet)
           mode: auto   # auto|random|native(expect .id to contain valid UUID string)
           extras:      # List of extra keys from properties to include (mode=auto)
           - "s2:generation_time"

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

    dss = stac2ds(items, stac_cfg, product_cache=product_cache)

    if patch_url is not None:
        dss = map(partial(patch_urls, edit=patch_url, bands=bands), dss)

    return dc_load(
        dss,
        measurements=bands,
        groupby=groupby,
        resampling=resampling,
        chunks=chunks,
        **geo,
        progress_cbk=progress_cbk,
        skip_broken_datasets=skip_broken_datasets,
        fuse_func=fuse_func,
        **kw,
    )
