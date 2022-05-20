"""
Various Dask helpers.
"""
import odc.geo.crs
import odc.geo.geobox
from dask.base import normalize_token


# TODO: these classes should just implement __dask_token__ instead
@normalize_token.register(odc.geo.crs.CRS)
def normalize_token_crs(crs):
    return ("odc.geo.crs.CRS", str(crs))


@normalize_token.register(odc.geo.geobox.GeoBox)
def normalize_token_geobox(gbox):
    crs = gbox.crs
    return ("odc.geo.geobox.GeoBox", str(crs), *gbox.shape.yx, *gbox.affine[:6])


@normalize_token.register(odc.geo.geobox.GeoboxTiles)
def normalize_token_gbt(gbt: odc.geo.geobox.GeoboxTiles):
    gbox = gbt.base
    crs = gbox.crs
    return (
        "odc.geo.geobox.GeoboxTiles",
        *gbt.shape.yx,
        str(crs),
        *gbox.shape.yx,
        *gbox.affine[:6],
    )
