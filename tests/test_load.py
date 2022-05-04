from unittest.mock import MagicMock

import pystac
import pystac.item
import pytest
import shapely.geometry
from odc.geo.geobox import GeoBox
from odc.geo.xr import ODCExtension

from odc.stac._load import _group_items
from odc.stac._load import load as stac_load
from odc.stac._model import ParsedItem
from odc.stac.testing.stac import b_, mk_parsed_item


def test_stac_load_smoketest(sentinel_stac_ms_with_raster_ext: pystac.item.Item):
    item = sentinel_stac_ms_with_raster_ext.clone()

    params = dict(crs="EPSG:3857", resolution=100, align=0, chunks={})
    with pytest.warns(UserWarning, match="`rededge`"):
        xx = stac_load([item], "B02", **params)

    assert isinstance(xx.B02.odc, ODCExtension)
    assert xx.B02.shape[0] == 1
    assert xx.B02.odc.geobox.crs == "EPSG:3857"

    # Test dc.load name for bands, and alias support
    with pytest.warns(UserWarning, match="`rededge`"):
        xx = stac_load([item], measurements=["red", "green"], **params)

    assert "red" in xx.data_vars
    assert "green" in xx.data_vars
    assert xx.red.shape == xx.green.shape

    # Test dc.load name for bands, and alias support
    patch_url = MagicMock(return_value="https://example.com/f.tif")
    xx = stac_load(
        [item],
        measurements=["red", "green"],
        patch_url=patch_url,
        stac_cfg={"*": {"warnings": "ignore"}},
        **params,
    )
    assert isinstance(xx.odc, ODCExtension)

    # expect patch_url to be called 2 times, 1 for red and 1 for green band
    assert patch_url.call_count == 2

    patch_url = MagicMock(return_value="https://example.com/f.tif")
    zz = stac_load(
        [item],
        patch_url=patch_url,
        stac_cfg={"*": {"warnings": "ignore"}},
        **params,
    )
    assert patch_url.call_count == len(zz.data_vars)

    yy = stac_load(
        [item], ["nir"], like=xx, chunks={}, stac_cfg={"*": {"warnings": "ignore"}}
    )
    assert yy.odc.geobox == xx.odc.geobox

    yy = stac_load(
        [item],
        ["nir"],
        geobox=xx.odc.geobox,
        chunks={},
        stac_cfg={"*": {"warnings": "ignore"}},
    )
    assert yy.odc.geobox == xx.odc.geobox
    assert yy.odc.geobox == yy.nir.odc.geobox

    # Check automatic CRS/resolution
    yy = stac_load(
        [item],
        ["nir", "coastal"],
        chunks={},
        stac_cfg={"*": {"warnings": "ignore"}},
    )
    assert yy.odc.geobox.crs == "EPSG:32606"
    assert yy.odc.geobox.resolution.yx == (-10, 10)

    # test bbox overlaping with lon/lat
    with pytest.raises(ValueError):
        stac_load([item], ["nir"], bbox=(0, 0, 1, 1), lon=(0, 1), lat=(0, 1), chunks={})

    # test bbox overlaping with x/y
    with pytest.raises(ValueError):
        stac_load(
            [item],
            ["nir"],
            bbox=(0, 0, 1, 1),
            x=(0, 1000),
            y=(0, 1000),
            chunks={},
        )

    bbox = (0, 0, 1, 1)
    x1, y1, x2, y2 = bbox

    assert (
        stac_load(
            [item],
            ["nir"],
            crs="epsg:3857",
            resolution=10,
            chunks={},
            lon=(x1, x2),
            lat=(y1, y2),
        ).nir.odc.geobox
        == stac_load(
            [item],
            ["nir"],
            crs="epsg:3857",
            resolution=10,
            chunks={},
            bbox=bbox,
        ).nir.odc.geobox
    )

    geopolygon = shapely.geometry.box(*bbox)
    assert (
        stac_load(
            [item],
            ["nir"],
            crs="epsg:3857",
            resolution=10,
            chunks={},
            lon=(x1, x2),
            lat=(y1, y2),
        ).nir.odc.geobox
        == stac_load(
            [item],
            ["nir"],
            crs="epsg:3857",
            resolution=10,
            chunks={},
            geopolygon=geopolygon,
        ).nir.odc.geobox
    )


def test_group_items():
    def _mk(id: str, lon: float, datetime: str):
        gbox = GeoBox.from_bbox([lon - 0.1, 0, lon + 0.1, 1], shape=(100, 100))
        return mk_parsed_item([b_("b1", gbox)], datetime=datetime, id=id)

    # check no-op case first
    assert _group_items([], "time") == []
    assert _group_items([], "nothing") == []
    assert _group_items([], "solar_day") == []

    aa = _mk("a", 15 * 10, "2020-01-02T23:59Z")
    b1 = _mk("b1", 15 * 10 + 1, "2020-01-03T00:01Z")
    b2 = _mk("b2", 15 * 10 + 2, "2020-01-03T00:01Z")
    cc = _mk("c", 0, "2020-01-02T23:59Z")

    assert _group_items([aa, b1, b2], "nothing") == [[aa], [b1], [b2]]
    assert _group_items([aa, b2, b1], "nothing") == [[aa], [b1], [b2]]
    assert _group_items([b1, aa, b2], "nothing") == [[aa], [b1], [b2]]
    assert _group_items([cc, aa, b1, b2], "nothing") == [[aa], [cc], [b1], [b2]]

    assert _group_items([aa, b1, b2], "time") == [[aa], [b1, b2]]
    assert _group_items([b1, aa, b2], "time") == [[aa], [b1, b2]]
    assert _group_items([b2, aa, b1], "time") == [[aa], [b1, b2]]
    assert _group_items([aa, cc, b1, b2], "time") == [[aa, cc], [b1, b2]]

    assert _group_items([aa, b1, b2], "solar_day") == [[aa, b1, b2]]
    assert _group_items([b1, aa, b2], "solar_day") == [[aa, b1, b2]]
    assert _group_items([b2, aa, b1], "solar_day") == [[aa, b1, b2]]
    assert _group_items([aa, b1, b2, cc], "solar_day") == [[cc], [aa, b1, b2]]

    with pytest.raises(ValueError):
        _ = _group_items([aa], groupby="no-such-mode")
