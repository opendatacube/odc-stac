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

    product_label = item['id']
    product_name = properties['platform']
    region_code = None

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
            product_id = str(odc_uuid("sentinel-2_stac_process", "1.0.0", [product_label]))
    elif properties.get('platform') in LANDSAT_PLATFORMS:
        product_id = item['id']
        product_label = _product_label(item)
        product_name = properties.get('odc:product')
        region_code = properties.get('odc:region_code')

    return product_id, product_label, product_name, region_code


def _product_label(item):
    uri = None
    for link in item.get("links"):
        rel = link.get("rel")
        if rel and rel == "self":
            uri = link.get("href")
    return Path(uri).stem.replace(".stac-item", "")


def _get_stac_bands(item, default_grid='g10m'):
    bands = {}

    grids = {}

    assets = item['assets']

    for asset_name, asset in assets.items():
        # Ignore items that are not actual COGs
        if asset['type'] not in ['image/tiff; application=geotiff; profile=cloud-optimized']:
            continue

        transform = asset['proj:transform']
        grid = f'g{transform[0]:g}m'

        if grid not in grids:
            grids[grid] = {
                'shape': asset['proj:shape'],
                'transform': asset['proj:transform']
            }

        band_info = {
            'path': Path(asset['href']).name
        }

        if grid != default_grid:
            band_info['grid'] = grid

        bands[asset_name] = band_info

    if default_grid in grids:
        grids['default'] = grids[default_grid]
        del grids[default_grid]

    return bands, grids


# TODO: Created for testing Landsat C3. Need to merge this function into _get_stac_bands()
def _get_stac_bands_ls3(item, default_grid='default'):
    bands = {}

    grids = {}

    default_grid = "default_grid"

    shape = item.get('properties').get('proj:shape')
    transform = item.get('properties').get('proj:transform')
    if shape and transform:
        grids[default_grid] = {
            'shape': item.get('properties').get('proj:shape'),
            'transform': item.get('properties').get('proj:transform')
        }

    assets = item['assets']

    for asset_name, asset in assets.items():
        if 'data' in asset.get('roles', []):
            grid = asset_name

            _grid = {
                'shape': asset.get('proj:shape'),
                'transform': asset.get('proj:transform')
            }

            band_info = {
                'path': Path(asset['href']).name
            }

            if _grid != grids[default_grid]:
                band_info['grid'] = grid
                grids[grid] = _grid

            bands[asset_name] = band_info

    if grids.get(default_grid):
        grids['default'] = grids[default_grid]
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

    product_id, product_label, product_name, region_code = _stac_product_lookup(input_stac)

    bands, grids = _get_stac_bands(input_stac, default_grid='g10m')

    properties = input_stac['properties']
    epsg = properties['proj:epsg']
    native_crs = f"epsg:{epsg}"

    geometry = _geographic_to_projected(input_stac['geometry'], native_crs)

    stac_odc = {
        '$schema': 'https://schemas.opendatacube.org/dataset',
        'id': product_id,
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
