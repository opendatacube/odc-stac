"""
stac.load - dc.load from STAC Items
"""
from functools import partial
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Union
import numpy as np

import pystac
import xarray
from datacube.utils.geometry import GeoBox
from odc.index import patch_urls

from ._dcload import dc_load
from ._eo3 import ConversionConfig, stac2ds


# pylint: disable=too-many-arguments
def load(
    items: Iterable[pystac.Item],
    bands: Optional[Union[str, Sequence[str]]] = None,
    geobox: Optional[GeoBox] = None,
    groupby: Optional[str] = None,
    resampling: Optional[Union[str, Dict[str, str]]] = None,
    skip_broken_datasets: bool = False,
    chunks: Optional[Dict[str, int]] = None,
    progress_cbk: Optional[Callable[[int, int], Any]] = None,
    fuse_func: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    stac_cfg: Optional[ConversionConfig] = None,
    patch_url: Optional[Callable[[str], str]] = None,
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
           output_crs="EPSG:32606",
           resolution=(100, -100),
       )
       xx.red.plot.imshow(col="time");


    .. note::

       At the moment one must specify desired projection and resolution of
       the result. The plan is to make this choice automatic if not configured.


    **Parameters**

    :param items:
       Iterable of STAC :class:`~pystac.Item` to load

    :param bands:
       List of band names to load, defaults to All. Also accepts
       single band name as input

    :param geobox:
       Allows to specify exact region/resolution/projection

    :param groupby:
       Controls what items get placed in to the same pixel plane,
       supported values are "time" or "solar_day", default is "time"

    :param resampling:
       Controls resampling strategy, can be specified per band

    :param skip_broken_datasets:
       Continue processing when IO errors are encountered

    :param chunks:
       Rather than loading pixel data directly construct
       Dask backed arrays. ``chunks={'x': 2048, 'y': 2048}``

    :param progress_cbk:
       Get data loading progress via callback, ignored when
       constructing Dask arrays

    :param fuse_func:
       Provide custom function for fusing pixels from different
       sources into one pixel plane.

       The default behaviour is to use first observed valid pixel.
       Item timestamp is used to determine order, ``nodata`` is
       used to determine "valid".

    :param stac_cfg:
       Controls :class:`pystac.Item` ``->`` :class:`datacube.model.Dataset`
       conversion, mostly used to specify "missing" metadata like pixel
       data types.

       See :func:`odc.stac.stac2ds` and section below for more details.

    :param patch_url:
       Optionally transform url of every band before loading

    :return:
       :class:`xarray.Dataset` with requested bands populated

    **Complete Example Code**

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
           output_crs="EPSG:32606",
           resolution=(100, -100),
           patch_url=pc.sign,
       )
       xx.red.plot.imshow(col="time", size=8, aspect=1);


    **Example Config**

    Sample ``stac_cfg=`` parameter.

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

       :func:`~odc.stac.stac2ds`,
       :py:meth:`datacube.Datacube.load`
    """
    if bands is None:
        # dc.load name for bands is measurements
        bands = kw.pop("measurements", None)

    dss = stac2ds(items, stac_cfg)

    if patch_url is not None:
        dss = map(partial(patch_urls, edit=patch_url, bands=bands), dss)

    return dc_load(
        dss,
        measurements=bands,
        geobox=geobox,
        groupby=groupby,
        resampling=resampling,
        skip_broken_datasets=skip_broken_datasets,
        chunks=chunks,
        progress_cbk=progress_cbk,
        fuse_func=fuse_func,
        **kw,
    )
