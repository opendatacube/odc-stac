"""Wrapper for Datacube.load_data."""
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Union
from warnings import warn

import datacube.utils.geometry
import shapely.geometry
import xarray as xr
from datacube import Datacube
from datacube.api.core import output_geobox
from datacube.model import Dataset
from datacube.utils.geometry import GeoBox


def _geojson_to_shapely(xx: Any) -> shapely.geometry.base.BaseGeometry:
    _type = xx.get("type", None)

    if _type is None:
        raise ValueError("Not a valid GeoJSON")

    _type = _type.lower()
    if _type == "featurecollection":
        features = xx.get("features", [])
        if len(features) == 1:
            return shapely.geometry.shape(features[0]["geometry"])
        else:
            return shapely.geometry.GeometryCollection(
                [shapely.geometry.shape(feature["geometry"]) for feature in features]
            )
    elif _type == "feature":
        return shapely.geometry.shape(xx["geometry"])

    return shapely.geometry.shape(xx)


def _normalize_geometry(xx: Any) -> datacube.utils.geometry.Geometry:
    if isinstance(xx, shapely.geometry.base.BaseGeometry):
        return datacube.utils.geometry.Geometry(xx, "epsg:4326")

    if isinstance(xx, datacube.utils.geometry.Geometry):
        return xx

    if isinstance(xx, dict):
        return datacube.utils.geometry.Geometry(_geojson_to_shapely(xx), "epsg:4326")

    # GeoPandas
    _geo = getattr(xx, "__geo_interface__", None)
    _crs = getattr(xx, "crs", "epsg:4326")
    return datacube.utils.geometry.Geometry(_geojson_to_shapely(_geo), _crs)


# pylint: disable=too-many-arguments
def dc_load(
    datasets: Iterable[Dataset],
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
    """Load data given a collection of datacube.Dataset objects."""
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

    if "geopolygon" in geo_keys:
        geo_keys["geopolygon"] = _normalize_geometry(geo_keys["geopolygon"])

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
