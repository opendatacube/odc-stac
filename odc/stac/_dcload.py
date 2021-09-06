"""
Wrapper for Datacube.load_data

"""
from typing import (
    Any,
    Optional,
    Union,
    Dict,
    Callable,
    Sequence,
)
from warnings import warn
import xarray as xr

from datacube import Datacube
from datacube.model import Dataset
from datacube.utils.geometry import GeoBox
from datacube.api.core import output_geobox


def dc_load(
    datasets: Sequence[Dataset],
    measurements: Optional[Union[str, Sequence[str]]] = None,
    geobox: Optional[GeoBox] = None,
    groupby: Optional[str] = None,
    resampling: Optional[Union[str, Dict[str, str]]] = None,
    skip_broken_datasets: bool = False,
    chunks: Optional[Dict[str, int]] = None,
    progress_cbk: Optional[Callable[[int, int], Any]] = None,
    fuse_func=None,
    **kw,
) -> xr.Dataset:
    """
    Load data given a collection of datacube.Dataset objects.
    """
    datasets = list(datasets)
    assert len(datasets) > 0

    # dask_chunks is a backward-compatibility alias for chunks
    if chunks is None:
        chunks = kw.pop("dask_chunks", None)
    # group_by is a backward-compatibility alias for groupby
    if groupby is None:
        groupby = kw.pop("group_by", "time")
    # bands alias for measurements
    if measurements is None:
        measurements = kw.pop("bands", None)

    # extract all "output_geobox" inputs
    geo_keys = {
        k: kw.pop(k)
        for k in [
            "like",
            "geopolygon",
            "resolution",
            "output_crs",
            "crs",
            "align",
            "x",
            "y",
            "lat",
            "lon",
        ]
        if k in kw
    }

    ds = datasets[0]
    product = ds.type

    if geobox is None:
        geobox = output_geobox(
            grid_spec=product.grid_spec,
            load_hints=product.load_hints(),
            **geo_keys,
            datasets=datasets,
        )
    elif len(geo_keys):
        warn(f"Supplied 'geobox=' parameter aliases {list(geo_keys)} inputs")

    grouped = Datacube.group_datasets(datasets, groupby)
    mm = product.lookup_measurements(measurements)
    return Datacube.load_data(
        grouped,
        geobox,
        mm,
        resampling=resampling,
        fuse_func=fuse_func,
        dask_chunks=chunks,
        skip_broken_datasets=skip_broken_datasets,
        progress_cbk=progress_cbk,
        **kw,
    )
