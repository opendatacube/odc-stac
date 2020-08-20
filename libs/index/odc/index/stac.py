import math
from pathlib import Path

from datacube.utils.geometry import Geometry
from odc.index import odc_uuid


KNOWN_CONSTELLATIONS = [
    'sentinel-2'
]

LANDSAT_PLATFORMS = [
    'landsat-5', 'landsat-7', 'landsat-8'
]


def _stac_product_lookup(item):
    properties = item['properties']

    # Maybe this should be the default product_name
    constellation = properties.get('constellation')

    if constellation in KNOWN_CONSTELLATIONS:
        if constellation == 'sentinel-2':
            product_label = properties['sentinel:product_id']
            product_name = 's2_l2a'
            region_code = '{}{}{}'.format(
                str(properties['proj:epsg'])[-2:],
                properties['sentinel:latitude_band'],
                properties['sentinel:grid_square']
            )
            default_grid = "g10m"
    elif properties.get('platform') in LANDSAT_PLATFORMS:
        product_label = _product_label(item)
        product_name = properties.get('odc:product')
        region_code = properties.get('odc:region_code')
        default_grid = "g30m"
    else:
        product_label = item['id']
        product_name = properties['platform']
        region_code = None
        default_grid = "default"

    return product_label, product_name, region_code, default_grid


def _product_label(item):
    uri = None
    for link in item.get("links"):
        rel = link.get("rel")
        if rel and rel == "self":
            uri = link.get("href")
    return Path(uri).stem.replace(".stac-item", "")


def _get_stac_bands(item, default_grid):
    bands = {}

    grids = {}

    assets = item.get('assets')

    for asset_name, asset in assets.items():
        # Ignore items that are not actual COGs/geotiff
        if asset.get('type') not in ['image/tiff; application=geotiff; profile=cloud-optimized',
                                 'image/tiff; application=geotiff']:
            continue

        transform = asset.get('proj:transform')
        grid = f'g{transform[0]:g}m'

        if grid not in grids:
            grids[grid] = {
                'shape': asset.get('proj:shape'),
                'transform': asset.get('proj:transform')
            }

        band_info = {
            'path': Path(asset.get('href')).name
        }

        if grid != default_grid:
            band_info['grid'] = grid

        bands[asset_name] = band_info

    if default_grid in grids:
        grids['default'] = grids.get(default_grid)
        del grids[default_grid]

    return bands, grids


def _geographic_to_projected(geometry, crs):
    """ Transform from WGS84 to the target projection, assuming Lon, Lat order
    """

    geom = Geometry(geometry, 'EPSG:4326')
    geom = geom.to_crs(crs, resolution=math.inf)

    if geom.is_valid:
        return geom.json
    else:
        return None


def stac_transform(input_stac):
    """ Takes in a raw STAC 1.0 dictionary and returns an ODC dictionary
    """

    product_label, product_name, region_code, default_grid = _stac_product_lookup(input_stac)

    # Generating UUID for products not having UUID.
    # TODO: Check is based on hardcoded Product Name, find generic way
    if product_name in ["s2_l2a"]:
        deterministic_uuid = str(odc_uuid("sentinel-2_stac_process", "1.0.0", [product_label]))
    else:
        deterministic_uuid = input_stac["id"]

    bands, grids = _get_stac_bands(input_stac, default_grid)

    properties = input_stac['properties']
    epsg = properties['proj:epsg']
    native_crs = f"epsg:{epsg}"

    geometry = _geographic_to_projected(input_stac['geometry'], native_crs)

    stac_odc = {
        '$schema': 'https://schemas.opendatacube.org/dataset',
        'id': deterministic_uuid,
        'crs': native_crs,
        'grids': grids,
        'product': {
            'name': product_name.lower()
        },
        'label': product_label,
        'properties': {
            'datetime': properties['datetime'].replace("000+00:00", "Z"),
            'odc:processing_datetime': properties['datetime'].replace("000+00:00", "Z"),
            'eo:cloud_cover': properties['eo:cloud_cover'],
            'eo:gsd': properties['gsd'],
            'eo:instrument': properties['instruments'][0],
            'eo:platform': properties['platform'],
            'odc:file_format': 'GeoTIFF'
        },
        'measurements': bands,
        'lineage': {}
    }

    if region_code:
        stac_odc['properties']['odc:region_code'] = region_code

    if geometry:
        stac_odc['geometry'] = geometry

    return stac_odc
