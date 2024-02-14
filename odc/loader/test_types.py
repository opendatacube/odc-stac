# pylint: disable=protected-access,missing-function-docstring,missing-module-docstring
import json

import pytest
from odc.geo.geobox import GeoBox

from .types import RasterBandMetadata, RasterLoadParams, RasterSource

gbox_4326 = GeoBox.from_bbox((103, -44, 169, -11), 4326, shape=200)
gbox_3857 = gbox_4326.to_crs(3857)


@pytest.mark.parametrize(
    "xx",
    [
        RasterLoadParams(),
        RasterSource("file:///tmp/x.tif"),
        RasterSource("file:///tmp/x.nc", subdataset="x"),
        RasterSource("x", meta=RasterBandMetadata("float32", -9999)),
        RasterSource("x", geobox=gbox_4326, meta=RasterBandMetadata("float32", -9999)),
        RasterSource("x", geobox=gbox_3857, meta=RasterBandMetadata("float32", -9999)),
    ],
)
def test_repr_json_smoke(xx):
    dd = xx._repr_json_()
    assert isinstance(dd, dict)
    assert json.dumps(dd)

    gbox = getattr(xx, "geobox", None)
    if gbox is not None:
        assert "crs" in dd
        assert "transform" in dd
        assert "shape" in dd
        assert dd["shape"] == gbox.shape.yx
        assert dd["crs"] == str(gbox.crs)
        assert dd["transform"] == list(gbox.transform)[:6]

    meta = getattr(xx, "meta", None)
    if meta is not None:
        assert "data_type" in dd
        assert "nodata" in dd
        assert dd["data_type"] == meta.data_type
        assert dd["nodata"] == meta.nodata
