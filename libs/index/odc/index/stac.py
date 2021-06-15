import math
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
from uuid import UUID

from datacube.utils.geometry import Geometry
from odc.index import odc_uuid
from toolz import get_in

Document = Dict[str, Any]

KNOWN_CONSTELLATIONS = ["sentinel-2"]

LANDSAT_PLATFORMS = ["landsat-5", "landsat-7", "landsat-8"]

# Mapping between EO3 field names and STAC properties object field names
MAPPING_STAC_TO_EO3 = {
    "end_datetime": "dtr:end_datetime",
    "start_datetime": "dtr:start_datetime",
    "gsd": "eo:gsd",
    "instruments": "eo:instrument",
    "platform": "eo:platform",
    "constellation": "eo:constellation",
    "view:off_nadir": "eo:off_nadir",
    "view:azimuth": "eo:azimuth",
    "view:sun_azimuth": "eo:sun_azimuth",
    "view:sun_elevation": "eo:sun_elevation",
}


def _stac_product_lookup(item: Document) -> Tuple[str, str, Optional[str], str]:
    properties = item["properties"]
    platform = properties.get("eo:platform", properties.get("platform", None))

    product_label = item["id"]
    product_name = get_in(["odc:product"], properties, platform)
    region_code = get_in(["odc:region_code"], properties, None)
    default_grid = None

    # Maybe this should be the default product_name
    constellation = properties.get("constellation")

    if constellation in KNOWN_CONSTELLATIONS:
        if constellation == "sentinel-2":
            product_label = properties["sentinel:product_id"]
            product_name = "s2_l2a"
            region_code = "{}{}{}".format(
                str(properties["proj:epsg"])[-2:],
                properties["sentinel:latitude_band"],
                properties["sentinel:grid_square"],
            )
            default_grid = "g10m"
    elif properties.get("platform") in LANDSAT_PLATFORMS:
        self_href = _find_self_href(item)
        product_label = Path(self_href).stem.replace(".stac-item", "")
        product_name = properties.get("odc:product")
        region_code = properties.get("odc:region_code")
        default_grid = "g30m"

    return product_label, product_name, region_code, default_grid


def _find_self_href(item: Document) -> str:
    """
    Extracting product label from filename of the STAC document 'self' URL
    """
    self_uri = [
        link.get("href", "")
        for link in item.get("links", [])
        if link.get("rel") == "self"
    ]

    if len(self_uri) < 1:
        raise ValueError("Can't find link for 'self'")
    if len(self_uri) > 1:
        raise ValueError("Too many links to 'self'")
    return self_uri[0]


def _get_stac_bands(
    item: Document, default_grid: str, relative: bool = False,
    proj_shape: str = None, proj_transform: str = None,
) -> Tuple[Document, Document]:
    bands = {}
    grids = {}
    assets = item.get("assets", {})

    for asset_name, asset in assets.items():
        # Ignore items that are not actual COGs/geotiff
        if asset.get("type") not in [
            "image/tiff; application=geotiff; profile=cloud-optimized",
            "image/tiff; application=geotiff",
        ]:
            continue

        transform = asset.get("proj:transform")
        if transform and proj_transform and proj_transform != transform:
            raise ValueError(
                'Conflicting proj:transform specified: {} and {}'.format(
                    transform, proj_transform,
                ))
        transform = transform if transform else proj_transform
        grid = f"g{transform[0]:g}m"

        shape = asset.get("proj:shape")
        if shape and proj_shape and shape != proj_shape:
            raise ValueError(
                'Conflicting proj:shape specified: {} and {}'.format(
                    shape, proj_shape,
                ))
        shape = shape if shape else proj_shape

        if grid not in grids:
            grids[grid] = {
                "shape": shape,
                "transform": transform,
            }

        path = asset["href"]
        band_index = asset.get("band", None)
        if relative:
            path = Path(path).name

        band_info = {"path": path}
        if band_index is not None:
            band_info["band"] = band_index

        # If we don't specify a default grid, label the first grid 'default'
        if not default_grid:
            default_grid = list(grids.keys())[0]

        if grid != default_grid:
            band_info["grid"] = grid

        bands[asset_name] = band_info

    if default_grid in grids:
        grids["default"] = grids.pop(default_grid)

    return bands, grids


def _geographic_to_projected(geometry, crs, precision=10):
    """Transform from WGS84 to the target projection, assuming Lon, Lat order"""
    geom = geometry.to_crs(crs, resolution=math.inf)

    def round_coords(c1, c2):
        return [round(coord, precision) for coord in [c1, c2]]

    if geom.is_valid:
        return geom.transform(round_coords)
    else:
        return None


def stac_transform_absolute(input_stac):
    return stac_transform(input_stac, relative=False)


def _convert_value_to_eo3_type(key: str, value):
    """
    Convert return type as per EO3 specification.
    Return type is String for "instrument" field in EO3 metadata.

    """
    if key == "instruments":
        if len(value) > 0:
            return "_".join([i.upper() for i in value])
        else:
            return None
    else:
        return value


def _get_stac_properties_lineage(input_stac: Document) -> Tuple[Document, Any]:
    """
    Extract properties and lineage field
    """
    properties = input_stac["properties"]
    prop = {
        MAPPING_STAC_TO_EO3.get(key, key): _convert_value_to_eo3_type(key, val)
        for key, val in properties.items()
    }
    if prop.get("odc:processing_datetime") is None:
        prop["odc:processing_datetime"] = properties["datetime"].replace(
            "000+00:00", "Z"
        )
    if prop.get("odc:file_format") is None:
        prop["odc:file_format"] = "GeoTIFF"

    # Extract lineage
    lineage = prop.pop("odc:lineage", None)

    return prop, lineage


def _check_valid_uuid(uuid_string: str) -> bool:
    """
    Check if provided uuid string is a valid UUID.
    """
    try:
        UUID(str(uuid_string))
        return True
    except ValueError:
        return False


def stac_transform(input_stac: Document, relative: bool = True) -> Document:
    """Takes in a raw STAC 1.0 dictionary and returns an ODC dictionary"""

    product_label, product_name, region_code, default_grid = _stac_product_lookup(
        input_stac
    )

    # Generating UUID for products not having UUID.
    # Checking if provided id is valid UUID.
    # If not valid, creating new deterministic uuid using odc_uuid function based on product_name and product_label.
    # TODO: Verify if this approach to create UUID is valid.
    if _check_valid_uuid(input_stac["id"]):
        deterministic_uuid = input_stac["id"]
    else:
        if product_name in ["s2_l2a"]:
            deterministic_uuid = str(
                odc_uuid("sentinel-2_stac_process", "1.0.0", [product_label])
            )
        else:
            deterministic_uuid = str(
                odc_uuid(f"{product_name}_stac_process", "1.0.0", [product_label])
            )

    # Check for projection extension properties that are not in the asset fields.
    # Specifically, proj:shape and proj:transform, as these are otherwise
    # fetched in _get_stac_bands.
    properties = input_stac["properties"]
    proj_shape = properties.get('proj:shape')
    proj_transform = properties.get('proj:transform')
    # TODO: handle old STAC that doesn't have grid information here...
    bands, grids = _get_stac_bands(
        input_stac, default_grid, relative=relative,
        proj_shape=proj_shape, proj_transform=proj_transform)

    stac_properties, lineage = _get_stac_properties_lineage(input_stac)

    epsg = properties["proj:epsg"]
    native_crs = f"epsg:{epsg}"

    # Transform geometry to the native CRS at an appropriate precision
    geometry = Geometry(input_stac["geometry"], "epsg:4326")
    if native_crs != "epsg:4326":
        # Arbitrary precisions, but should be fine
        pixel_size = get_in(["default", "transform", 0], grids)
        precision = 0
        if pixel_size < 0:
            precision = 6

        geometry = _geographic_to_projected(geometry, native_crs, precision)

    stac_odc = {
        "$schema": "https://schemas.opendatacube.org/dataset",
        "id": deterministic_uuid,
        "crs": native_crs,
        "grids": grids,
        "product": {"name": product_name.lower()},
        "label": product_label,
        "properties": stac_properties,
        "measurements": bands,
        "lineage": {},
    }

    if region_code:
        stac_odc["properties"]["odc:region_code"] = region_code

    if geometry:
        stac_odc["geometry"] = geometry.json

    if lineage:
        stac_odc["lineage"] = lineage

    return stac_odc
