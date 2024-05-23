# pylint: disable=protected-access,missing-function-docstring,missing-module-docstring
import json

import pytest
from odc.geo.geobox import GeoBox

from .types import (
    FixedCoord,
    RasterBandMetadata,
    RasterGroupMetadata,
    RasterLoadParams,
    RasterSource,
    with_default,
)

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
        RasterGroupMetadata({}),
        RasterGroupMetadata(
            bands={("x", 1): RasterBandMetadata("float32", -9999)},
            aliases={"X": [("x", 1)]},
            extra_dims={"b": 3},
            extra_coords=[
                FixedCoord("b", ["a", "b", "c"]),
                FixedCoord("B", [1, 2, 3], dtype="int32", dim="b"),
            ],
        ),
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


def test_with_default():
    A = object()
    B = "B"
    assert with_default(None, A) is A
    assert with_default(A, B) is A
    assert with_default(A, B, A) is B
    assert with_default((), B, (), {}) is B
