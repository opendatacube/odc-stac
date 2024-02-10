# pylint: disable=missing-function-docstring, missing-module-docstring
from dask.base import tokenize
from odc.geo.geobox import CRS, GeoBox, GeoboxTiles

from ._dask import tokenize_stream


def test_tokenize_odc_geo():
    gbox = GeoBox.from_bbox([0, 0, 1, 1], shape=(100, 100))
    assert tokenize(gbox) == tokenize(gbox)
    assert tokenize(gbox) != tokenize(gbox.pad(1))

    gbt = GeoboxTiles(gbox, (1, 3))
    assert tokenize(gbt) == tokenize(gbt)
    assert tokenize(GeoboxTiles(gbox, (1, 2))) != tokenize(GeoboxTiles(gbox, (2, 1)))
    assert tokenize(GeoboxTiles(gbox, (1, 2))) != tokenize(
        GeoboxTiles(gbox.pad(1), (1, 2))
    )

    crs = CRS("epsg:4326")
    assert tokenize(crs) == tokenize(crs)
    assert tokenize(crs) == tokenize(CRS(crs))
    assert tokenize(CRS("epsg:4326")) == tokenize(CRS("EPSG:4326"))
    assert tokenize(CRS("epsg:4326")) != tokenize(CRS("EPSG:3857"))


def test_tokenize_stream():
    dd = [1, 2, "", 2]
    kx = list(tokenize_stream(dd))
    assert kx == [(tokenize(d), d) for d in dd]

    kx = list(tokenize_stream(dd, lambda k: (k, "d")))
    assert kx == [((tokenize(d), "d"), d) for d in dd]

    dsk = {}
    kx = list(tokenize_stream(dd, None, dsk))
    assert len(dsk) == len(set(dd))

    for k, x in kx:
        assert dsk[k] is x
