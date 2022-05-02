"""stac.load - dc.load from STAC Items."""
import dataclasses
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, Union, cast

import numpy as np
import pystac
import pystac.item
import xarray as xr
from dask import array as da
from odc.geo import XY, MaybeCRS, SomeResolution
from odc.geo.geobox import GeoBox
from odc.geo.xr import wrap_xr

from ._mdtools import ConversionConfig, output_geobox, parse_items
from ._model import ParsedItem, RasterCollectionMetadata


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
    # pylint: disable=unused-argument
    if bands is None:
        # dc.load name for bands is measurements
        bands = kw.pop("measurements", None)

    # normalize args
    # dc.load compatible name for crs is `output_crs`
    if crs is None:
        crs = cast(MaybeCRS, kw.pop("output_crs", None))

    parsed_items = list(parse_items(items, cfg=stac_cfg))

    if patch_url is not None:
        parsed_items = [
            patch_urls(item, edit=patch_url, bands=bands) for item in parsed_items
        ]

    gbox = output_geobox(
        parsed_items,
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

    # use dummy for now we'll test geobox, groupby and time dimension with that first
    nt = len(parsed_items)

    def _dummy(dtype):
        _shape = (nt, *gbox.shape.yx)
        if chunks is None:
            return np.zeros(_shape, dtype=dtype)

        _chunks = (1, *tuple(chunks.get(dim, -1) for dim in gbox.dimensions))
        return da.zeros(_shape, dtype=dtype, chunks=_chunks)

    collection = _collection(parsed_items)
    mm = collection.resolve_bands(bands)
    data_bands = {
        name: wrap_xr(_dummy(m.data_type), gbox, nodata=m.nodata)
        for name, m in mm.items()
    }

    return xr.Dataset(data_bands)  # type: ignore
