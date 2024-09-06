import datetime
import math
from pathlib import Path
from typing import Iterable, Iterator
import mimetypes

import pystac.asset
import pystac.collection
import pystac.errors
import pystac.item
from pystac.extensions.eo import Band, EOExtension
from pystac.extensions.projection import ProjectionExtension
from pystac.extensions.view import ViewExtension
from pystac import Asset, Link, MediaType
from pystac.utils import datetime_to_str
from pystac.errors import STACError

from datacube.model import Dataset
from datacube.utils.uris import uri_resolve
from datacube.index.eo3 import EO3Grid

from ._eo3converter import STAC_TO_EO3_RENAMES

MAPPING_EO3_TO_STAC = {v: k for k, v in STAC_TO_EO3_RENAMES.items()}

def _as_stac_instruments(value: str):
    """
    >>> _as_stac_instruments('TM')
    ['tm']
    >>> _as_stac_instruments('OLI')
    ['oli']
    >>> _as_stac_instruments('ETM+')
    ['etm']
    >>> _as_stac_instruments('OLI_TIRS')
    ['oli', 'tirs']
    """
    return [i.strip("+-").lower() for i in value.split("_")]


# may need to be more robust
def _convert_value_to_stac_type(key: str, value):
    """
    Convert return type as per STAC specification
    """
    # In STAC spec, "instruments" have [String] type
    if key == "eo:instrument":
        return _as_stac_instruments(value)
    # Convert the non-default datetimes to a string
    elif isinstance(value, datetime.datetime) and key != "datetime":
        return datetime_to_str(value)
    else:
        return value


def _media_type(path: Path) -> str:
    """
    Add media type of the asset object
    """
    mime_type = mimetypes.guess_type(path.name)[0]
    if path.suffix == ".sha1":
        return MediaType.TEXT
    elif path.suffix == ".yaml":
        return "text/yaml"
    elif mime_type:
        if mime_type == "image/tiff":
            return MediaType.COG
        else:
            return mime_type
    else:
        return "application/octet-stream"


def _asset_roles_fields(asset_name: str) -> list[str]:
    """
    Add roles of the asset object
    """
    if asset_name.startswith("thumbnail"):
        return ["thumbnail"]
    else:
        return ["metadata"]


def _asset_title_fields(asset_name: str) -> str | None:
    """
    Add title of the asset object
    """
    if asset_name.startswith("thumbnail"):
        return "Thumbnail image"
    else:
        return None


def _proj_fields(grids: dict[str, EO3Grid], grid_name: str = "default") -> dict:
    """
    Get any proj (Stac projection extension) fields if we have them for the grid.
    """
    if not grids:
        return {}

    grid_doc = grids.get(grid_name or "default")
    if not grid_doc:
        return {}

    return {
        "shape": grid_doc.shape,
        "transform": grid_doc.transform,
    }


def _lineage_fields(dataset: Dataset) -> dict:
    """
    Add custom lineage field to a STAC Item
    """
    dataset_doc = dataset.metadata_doc
    # using sources vs source_tree?
    if dataset.sources:
        lineage = {classifier: [str(d.id)] for classifier, d in dataset.sources}
    elif dataset_doc.get("lineage"):
        # sometimes lineage is included at 'lineage' instead of 'lineage.source_datasets'
        # in which case it should already be in {classifier: [ids]} format
        lineage = dataset_doc.get("lineage")
        # shouldn't need to account for legacy embedded lineage at this point
    else:
        return {}
    # it seems like derived are not accounted for at all - on purpose?
    
    return {"odc:lineage": lineage}


def eo3_to_stac_properties(dataset: Dataset) -> dict:
    """
    Convert EO3 properties dictionary to the Stac equivalent.
    """
    title = dataset.metadata.label
    # explorer has logic to try and figure out a label if missing, should it be included here?
    properties = {
        # Put the title at the top for document readability.
        **(dict(title=title) if title else {}),
        **{
            MAPPING_EO3_TO_STAC.get(key, key): _convert_value_to_stac_type(key, val)
            for key, val in dataset.metadata_doc.properties.items()
        },
    }

    return properties


def ds_to_item(
    dataset: Dataset,
    stac_item_url: str | None = None,
) -> pystac.Item:
    """
    Convert the given ODC Dataset into a Stac Item document.

    Note: You may want to call `validate_item(doc)` on the outputs to find any
    incomplete properties.

    :param collection_url: URL to the Stac Collection. Either this or an explorer_base_url
                           should be specified for Stac compliance.
    :param stac_item_url: Public 'self' URL where the stac document will be findable.
    """
    if not dataset.is_eo3:
        raise STACError("Cannot convert non-eo3 datasets to STAC")

    if stac_item_url is None and not dataset.uri:
        raise ValueError("No dataset location provided")

    if dataset.extent is not None:
        wgs84_geometry = dataset.extent.to_crs("EPSG:4326", math.inf)
        geometry = wgs84_geometry.json
        bbox = wgs84_geometry.boundingbox.bbox
    else:
        geometry = None
        bbox = None

    properties = eo3_to_stac_properties(dataset)
    properties.update(_lineage_fields(dataset))

    dt = dataset.time
    if dt is None:
        raise ValueError("Cannot convert dataset with no datetime information")
    dt_info = {}
    if dt.begin == dt.end:
        dt_info["start_datetime"] = dt.begin
        dt_info["end_datetime"] = dt.end
    else:
        dt_info["datetime"] = dt.begin
        properties.pop("datetime", None)

    item = pystac.item.Item(
        id=str(dataset.id),
        **dt_info,
        properties=properties,
        geometry=geometry,
        bbox=bbox,
        collection=dataset.product.name,
    )

    EOExtension.ext(item, add_if_missing=True)

    # while dataset._gs has already handled grid information, it doesn't allow us to 
    # access all the information we need here
    grids = {name: EO3Grid(grid_spec) for name, grid_spec in dataset.metadata_doc.get("grids", {}).items()}

    if geometry:
        proj = ProjectionExtension.ext(item, add_if_missing=True)
        
        if dataset.crs is None:
            raise STACError("Projection extension requires either epsg or wkt for crs.")
        if dataset.crs.epsg is not None:
            proj.apply(epsg=dataset.crs.epsg, **_proj_fields(grids))
        else:
            proj.apply(wkt2=dataset.crs, **_proj_fields(grids))

    # To pass validation, only add 'view' extension when we're using it somewhere.
    if any(k.startswith("view:") for k in properties.keys()):
        ViewExtension.ext(item, add_if_missing=True)

    # Without a dataset location, all paths will be relative.
    dataset_location = dataset.uri # or should we default to stac_url?

    # Add assets that are data
    for name, measurement in dataset.measurements.items():
        if not dataset_location and not measurement.get("path"):
            # No URL to link to. URL is mandatory for Stac validation.
            continue

        asset = Asset(
            href=uri_resolve(dataset_location, measurement.get("path")),
            media_type=_media_type(Path(measurement.get("path"))),
            title=name,
            roles=["data"],
        )
        eo = EOExtension.ext(asset)

        # TODO: pull out more information about the band
        band = Band.create(name)
        eo.apply(bands=[band])

        if grids:
            proj_fields = _proj_fields(grids, measurement.get("grid"))
            if proj_fields is not None:
                proj = ProjectionExtension.ext(asset)
                # Not sure how this handles None for an EPSG code
                # should we have a wkt2 case like above?
                proj.apply(
                    **proj_fields,
                    epsg=dataset.crs.epsg,
                )

        item.add_asset(name, asset=asset)

    # Add assets that are accessories
    for name, acc in dataset.accessories.items():
        if not dataset_location and not acc.get("path"):
            # No URL to link to. URL is mandatory for Stac validation.
            continue

        asset = Asset(
            href=uri_resolve(dataset_location, acc.get("path")),
            media_type=_media_type(Path(acc.get("path"))),
            title=_asset_title_fields(name),
            roles=_asset_roles_fields(name),
        )

        item.add_asset(name, asset=asset)
    
    # should all item links be handled externally?
    if stac_item_url:
        item.links.append(
            Link(
                rel="self",
                media_type=MediaType.JSON,
                target=stac_item_url,
            )
        )

    return item


def ds2stac(datasets: Iterable[Dataset]) -> Iterator[pystac.item.Item]:
    for dataset in datasets:
        yield ds_to_item(dataset, dataset.uri)
