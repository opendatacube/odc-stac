from unittest.mock import MagicMock

import pystac
import pystac.item
import pytest
import shapely.geometry
from dask.utils import ndeepmap
from odc.geo.geobox import GeoBox
from odc.geo.xr import ODCExtension

from odc.loader import resolve_load_cfg
from odc.stac import RasterLoadParams
from odc.stac import load as stac_load
from odc.stac._stac_load import _group_items
from odc.stac.testing.stac import b_, mk_parsed_item, to_stac_item


def test_stac_load_smoketest(sentinel_stac_ms_with_raster_ext: pystac.item.Item):
    item = sentinel_stac_ms_with_raster_ext.clone()

    params = dict(crs="EPSG:3857", resolution=100, align=0, chunks={})
    xx = stac_load([item], "B02", **params)

    assert isinstance(xx.B02.odc, ODCExtension)
    assert xx.B02.shape[0] == 1
    assert xx.B02.odc.geobox is not None
    assert xx.B02.odc.geobox.crs == "EPSG:3857"
    assert xx.time.dtype == "datetime64[ns]"

    # Test dc.load name for bands, and alias support
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
        gbox = GeoBox.from_bbox((lon - 0.1, 0, lon + 0.1, 1), shape=(100, 100))
        return mk_parsed_item([b_("b1", gbox)], datetime=datetime, id=id)

    # check no-op case first
    assert _group_items([], [], "time") == []
    assert _group_items([], [], "id") == []
    assert _group_items([], [], "solar_day") == []

    aa = _mk("a", 15 * 10, "2020-01-02T23:59Z")
    b1 = _mk("b1", 15 * 10 + 1, "2020-01-03T00:01Z")
    b2 = _mk("b2", 15 * 10 + 2, "2020-01-03T00:01Z")
    cc = _mk("c", 0, "2020-01-02T23:59Z")

    def _t(items, groupby, expect, lon=None, preserve_original_order=False):
        stac_items = [to_stac_item(item) for item in items]
        rr = ndeepmap(
            2,
            lambda idx: items[idx],
            _group_items(
                stac_items,
                items,
                groupby,
                lon=lon,
                preserve_original_order=preserve_original_order,
            ),
        )
        _expect = ndeepmap(2, lambda item: item.id, expect)
        _got = ndeepmap(2, lambda item: item.id, rr)

        assert _expect == _got

    # same order as input
    _t([aa, b1, b2], "id", [[aa], [b1], [b2]])
    _t([aa, b2, b1], "id", [[aa], [b2], [b1]])
    _t([b1, aa, b2], "id", [[b1], [aa], [b2]])
    _t([cc, aa, b1, b2], "id", [[cc], [aa], [b1], [b2]])

    _t([aa, b1, b2], "time", [[aa], [b1, b2]])
    _t([b1, aa, b2], "time", [[aa], [b1, b2]])

    # order within group is preserved
    _t([b2, aa, b1], "time", [[aa], [b2, b1]], preserve_original_order=True)
    _t([aa, cc, b1, b2], "time", [[aa, cc], [b1, b2]], preserve_original_order=True)

    _t([aa, b1, b2], "solar_day", [[aa, b1, b2]])
    _t([b1, aa, b2], "solar_day", [[aa, b1, b2]])
    _t([b2, aa, b1], "solar_day", [[aa, b1, b2]])
    _t([aa, b1, b2, cc], "solar_day", [[cc], [aa, b1, b2]])

    _t([aa, b1, b2, cc], "solar_day", [[aa, cc, b1, b2]], lon=150 + 1)

    # property based
    _t([aa, b1], "proj:epsg", [[aa, b1]])
    _t([b1, aa], "proj:epsg", [[aa, b1]])
    _t([aa, b1], "proj:transform", [[aa], [b1]])

    # custom callback
    _t(
        [aa, b1, b2, cc],
        lambda item, parsed, idx: idx % 2,
        [[aa, b2], [b1, cc]],
        preserve_original_order=True,
    )


def test_resolve_load_cfg():
    rlp = RasterLoadParams
    assert resolve_load_cfg({}) == {}

    item = mk_parsed_item(
        [
            b_("a", dtype="int8", nodata=-1),
            b_("b", dtype="float64"),
        ]
    )

    assert set(item.collection) == set([("a", 1), ("b", 1)])
    assert item.collection["a"].data_type == "int8"
    assert item.collection["b"].data_type == "float64"

    _bands = {n: b for (n, _), b in item.collection.bands.items()}

    cfg = resolve_load_cfg(_bands, resampling="average")
    assert cfg["a"] == rlp("int8", -1, resampling="average")
    assert cfg["b"] == rlp("float64", None, resampling="average")

    cfg = resolve_load_cfg(
        _bands,
        resampling={"*": "mode", "b": "sum"},
        nodata=-999,
        dtype="int64",
    )
    assert cfg["a"] == rlp("int64", -999, resampling="mode")
    assert cfg["b"] == rlp("int64", -999, resampling="sum")

    cfg = resolve_load_cfg(
        _bands,
        dtype={"a": "float32"},
    )
    assert cfg["a"] == rlp("float32", -1)
    assert cfg["b"] == rlp("float64", None)
