from affine import Affine
from datacube.utils.geometry import (
    CRS,
    Geometry,
    polygon_from_transform,
    bbox_union,
)


def grid2points(grid):
    h, w = grid['shape']
    transform = Affine(*grid['transform'][:6])
    pts = [(0, 0), (w, 0), (w, h), (0, h)]
    return [transform*pt for pt in pts]


def grid2ref_points(grid):
    nn = ['ul', 'ur', 'lr', 'll']
    return {n: dict(x=x, y=y)
            for n, (x, y) in zip(nn, grid2points(grid))}


def grid2polygon(grid, crs):
    h, w = grid['shape']
    transform = Affine(*grid['transform'][:6])

    if isinstance(crs, str):
        crs = CRS(crs)

    return polygon_from_transform(w, h, transform, crs)


def eo3_lonlat_bbox(doc, tol=None):
    epsg4326 = CRS('epsg:4326')
    crs = CRS(doc['crs'])
    grids = doc['grids']
    geometry = doc.get('geometry')
    if geometry is None:
        return bbox_union(grid2polygon(grid, crs).to_crs(epsg4326, tol).boundingbox
                          for grid in grids.values())
    else:
        return Geometry(geometry, crs).to_crs(epsg4326, tol).boundingbox


def eo3_grid_spatial(doc, tol=None):
    """
    Using doc[grids|crs|geometry] compute EO3 style grid spatial:

    ```
      extent:
        lat: {begin=<>, end=<>}
        lon: {begin=<>, end=<>}

      grid_spatial:
        projection:
          spatial_reference: "<crs>"
          geo_ref_points: {ll: {x:<>, y:<>}, ...}
          valid_data: {...}
    ```
    """
    grid = doc['grids']['default']
    crs = doc['crs']
    geometry = doc.get('geometry')

    if geometry is not None:
        valid_data = dict(valid_data=geometry)
    else:
        valid_data = {}

    oo = dict(grid_spatial=dict(projection={
        'spatial_reference': crs,
        'geo_ref_points': grid2ref_points(grid),
        **valid_data,
    }))

    bb = eo3_lonlat_bbox(doc, tol=tol)
    oo['extent'] = dict(
        lat=dict(begin=bb.bottom, end=bb.top),
        lon=dict(begin=bb.left, end=bb.right))

    return oo
