from affine import Affine
import toolz
from types import SimpleNamespace

from datacube.utils.geometry import (
    CRS,
    Geometry,
    polygon_from_transform,
    bbox_union,
)


def _norm_grid(grid):
    shape = grid.get('shape')
    transform = grid.get('transform')
    if shape is None or transform is None:
        raise ValueError("Each grid must have .shape and .transform")
    return SimpleNamespace(shape=shape,
                           transform=Affine(*transform[:6]))


def grid2points(grid):
    grid = _norm_grid(grid)

    h, w = grid.shape
    transform = grid.transform
    pts = [(0, 0), (w, 0), (w, h), (0, h)]
    return [transform*pt for pt in pts]


def grid2ref_points(grid):
    nn = ['ul', 'ur', 'lr', 'll']
    return {n: dict(x=x, y=y)
            for n, (x, y) in zip(nn, grid2points(grid))}


def grid2polygon(grid, crs):
    grid = _norm_grid(grid)
    h, w = grid.shape
    transform = grid.transform

    if isinstance(crs, str):
        crs = CRS(crs)

    return polygon_from_transform(w, h, transform, crs)


def eo3_lonlat_bbox(doc, tol=None):
    epsg4326 = CRS('epsg:4326')
    crs = doc.get('crs')
    grids = doc.get('grids')

    if crs is None or grids is None:
        raise ValueError("Input must have crs and grids")

    crs = CRS(crs)
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
    grid = toolz.get_in(['grids', 'default'], doc, None)
    crs = doc.get('crs', None)
    if crs is None or grid is None:
        raise ValueError("Input must have crs and grids.default")

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

def add_eo3_parts(doc, tol=None):
    """Add spatial keys the DB requires to eo3 metadata
    """
    return dict(**doc,
                **eo3_grid_spatial(doc, tol=tol))

def prep_eo3(doc, tol=None):
    """Modify spatial and lineage sections of eo3 metadata
    """
    if doc is None:
        return None

    doc = add_eo3_parts(doc, tol=tol)
    lineage = doc.pop('lineage', {})

    def remap_lineage(uuids):
        """ Turn [uuid] -> {id: uuid}
        """
        uuid, *_ = uuids
        return dict(id=uuid)

    doc['lineage'] = dict(source_datasets={
        n: remap_lineage(v) for n, v in lineage.items()})

    return doc

def detect_eo3(doc : dict) -> bool:
    """Heuristics to auto-detect and apply eo3 transforms
    
    Arguments:
        doc {dict} -- ODC Metadata (typically from YAML)
    
    Returns:
        bool -- is this metadata eo3 ?
    """
    return ('grids' in doc) and ('default' in doc['grids']) and ('transform' in doc['grids']['default'])