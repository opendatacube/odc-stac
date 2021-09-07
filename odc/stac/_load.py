"""
stac.load - dc.load from STAC Items
"""
from functools import partial
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Union
import numpy as np

import pystac
import xarray as xr
from datacube.model import Dataset
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
) -> xr.Dataset:
    """
    Load STAC Items (from the same or similar collections) as an
    xarray Dataset.

    :param items:
       STAC Items to load

    :param bands:
        List of band names to load, defaults to All. Also accepts
        single band name as input.

    :param geobox:
       Allows to specify exact region/resolution/projection

    :param groupby:
       Controls what items get placed in to the same pixel plane,
       supported values are "time" or "solar_day", default is "time"

    :param resampling:
       Controls resampling strategy, can be specified per band.

    :param skip_broken_datasets:
       Continue processing when IO errors are encountered

    :param chunks:
       Rather than loading pixel data directly construct
       Dask backed arrays. ``chunks={'x': 2048, 'y': 2048}``

    :param progress_cbk:
       Get data loading progress via callback, ignored when
       constructing Dask arrays.

    :param fuse_func:
       Provide custom function for fusing pixels from different
       sources into one pixel plane. The default behaviour is to
       use first observed valid pixel (Item timestamp is used to
       determine "first", ``nodata`` is used to determine "valid")

    :param stac_cfg:
       Controls STAC -> Dataset conversion, mostly used to specify
       "missing" metadata like pixel data types. see ``stac_to_ds``.

    :param patch_url:
       Optionally transform url of every band before loading

    :return:
       ``xr.Dataset`` with requested bands populated
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
