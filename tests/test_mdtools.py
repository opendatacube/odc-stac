# pylint: disable=redefined-outer-name,missing-module-docstring,missing-function-docstring,missing-class-docstring
# pylint: disable=use-implicit-booleaness-not-comparison,protected-access
from __future__ import annotations

import pystac
import pystac.asset
import pystac.collection
import pystac.item
import pystac.utils
import pytest
from common import NO_WARN_CFG, S2_ALL_BANDS, STAC_CFG
from odc.geo import geom
from odc.geo.geobox import AnchorEnum, GeoBox, geobox_union_conservative
from odc.geo.types import xy_
from odc.geo.xr import xr_zeros
from pystac.extensions.projection import ProjectionExtension

from odc.loader.testing.fixtures import FakeMDPlugin
from odc.loader.types import FixedCoord, RasterBandMetadata, RasterGroupMetadata
from odc.stac._mdtools import (
    _auto_load_params,
    _gbox_anchor,
    _most_common_gbox,
    _normalize_geometry,
    asset_geobox,
    band_metadata,
    compute_eo3_grids,
    extract_collection_metadata,
    has_proj_ext,
    has_raster_ext,
    is_raster_data,
    output_geobox,
    parse_item,
    parse_items,
)
from odc.stac.model import ParsedItem
from odc.stac.testing.stac import b_, mk_parsed_item, to_stac_item

GBOX = GeoBox.from_bbox((-20, -10, 20, 10), "epsg:3857", shape=(200, 400))


def test_is_raster_data(sentinel_stac_ms: pystac.item.Item):
    item = sentinel_stac_ms
    assert "B01" in item.assets
    assert "B02" in item.assets

    assert is_raster_data(item.assets["B01"])

    # check case when roles are missing
    item.assets["B02"].roles = None
    assert is_raster_data(item.assets["B02"])


def test_eo3_grids(sentinel_stac_ms: pystac.item.Item):
    item0 = sentinel_stac_ms

    item = item0.clone()
    assert item.collection_id == "sentinel-2-l2a"

    data_bands = {
        name: asset
        for name, asset in item.assets.items()
        if is_raster_data(asset, check_proj=True)
    }

    grids, b2g = compute_eo3_grids(data_bands)
    assert set(grids) == set("default g20 g60".split(" "))
    assert set(grids) == set(b2g.values())
    assert set(b2g) == set(data_bands)

    # test the case where there are different shapes for the same gsd
    # clashing grid names should be resolved
    ProjectionExtension.ext(item.assets["B01"]).shape = (100, 200)  # type: ignore
    grids, b2g = compute_eo3_grids(data_bands)
    assert b2g["B01"] != b2g["B02"]

    # More than 1 CRS should work
    item = item0.clone()
    ProjectionExtension.ext(item.assets["B01"]).epsg = 3857
    assert ProjectionExtension.ext(item.assets["B01"]).epsg == 3857

    data_bands = {k: item.assets[k] for k in data_bands}
    grids, b2g = compute_eo3_grids(data_bands)
    assert b2g["B01"] != b2g["B02"]
    crs = grids[b2g["B01"]].crs
    assert crs is not None
    assert crs.epsg == 3857


def test_asset_geobox(sentinel_stac: pystac.item.Item):
    item0 = sentinel_stac
    item = item0.clone()
    asset = item.assets["B01"]
    geobox = asset_geobox(asset)
    assert geobox.shape == (1830, 1830)

    # Tests non-affine transofrm ValueError
    item = item0.clone()
    asset = item.assets["B01"]
    ProjectionExtension.ext(asset).transform[-1] = 2  # type: ignore
    with pytest.raises(ValueError):
        asset_geobox(asset)

    # Tests wrong-sized transform transofrm ValueError
    item = item0.clone()
    asset = item.assets["B01"]
    ProjectionExtension.ext(asset).transform = [1, 1, 2]
    with pytest.raises(ValueError):
        asset_geobox(asset)

    # Test missing transform transofrm ValueError
    item = item0.clone()
    asset = item.assets["B01"]
    ProjectionExtension.ext(asset).transform = None
    with pytest.raises(ValueError):
        asset_geobox(asset)

    # Test no proj extension case
    item = item0.clone()
    item.stac_extensions = []
    asset = item.assets["B01"]
    with pytest.raises(ValueError):
        asset_geobox(asset)


def test_has_proj_ext(sentinel_stac_ms_no_ext: pystac.item.Item):
    assert has_proj_ext(sentinel_stac_ms_no_ext) is False


def test_band_metadata(sentinel_stac_ms_with_raster_ext: pystac.item.Item):
    item = sentinel_stac_ms_with_raster_ext.clone()
    assert has_raster_ext(item) is True
    asset = item.assets["SCL"]
    bm = band_metadata(asset, RasterBandMetadata("uint16", 0, "1"))
    assert bm == [RasterBandMetadata("uint8", 0, "1")]

    # Test multiple bands per asset produce multiple outputs
    asset.extra_fields["raster:bands"].append({"nodata": -10})
    bm = band_metadata(asset, RasterBandMetadata("uint16", 0, "1"))
    assert bm == [
        RasterBandMetadata("uint8", 0, "1"),
        RasterBandMetadata(data_type="uint16", nodata=-10, unit="1"),
    ]


def test_is_raster_data_more():
    def _a(href="http://example.com/", **kw):
        return pystac.asset.Asset(href, **kw)

    assert is_raster_data(_a(media_type="image/jpeg")) is True
    assert is_raster_data(_a(media_type="image/jpeg", roles=["data"])) is True
    assert is_raster_data(_a(media_type="image/jpeg", roles=["overview"])) is False
    assert is_raster_data(_a(media_type="image/jpeg", roles=["thumbnail"])) is False

    # no media type defined
    assert is_raster_data(_a(roles=["data"])) is True
    assert is_raster_data(_a(roles=["metadata"])) is False
    assert is_raster_data(_a(roles=["custom-22"])) is False

    # based on extension
    assert is_raster_data(_a(href="/foo.tif")) is True
    assert is_raster_data(_a(href="/foo.tiff")) is True
    assert is_raster_data(_a(href="/foo.TIF")) is True
    assert is_raster_data(_a(href="/foo.TIFF")) is True
    assert is_raster_data(_a(href="/foo.jpeg")) is True
    assert is_raster_data(_a(href="/foo.jpg")) is True


def test_extract_md(sentinel_stac_ms: pystac.item.Item):
    item0 = sentinel_stac_ms
    item = pystac.Item.from_dict(item0.to_dict())

    assert item.collection_id in STAC_CFG

    md = extract_collection_metadata(item, STAC_CFG)

    assert md.name == "sentinel-2-l2a"

    assert set(md.all_bands) == S2_ALL_BANDS

    # check defaults were set
    for b in ["B01", "B02", "B03"]:
        bk: tuple[str, int] = (b, 1)
        assert md.bands[bk].data_type == "uint16"
        assert md.bands[bk].nodata == 0
        assert md.bands[bk].unit == "1"

    assert md.bands[("SCL", 1)].data_type == "uint8"
    assert md.bands[("visual", 1)].data_type == "uint8"

    # check aliases configuration
    assert md.aliases["rededge"] == [("B05", 1), ("B06", 1), ("B07", 1), ("B8A", 1)]
    assert md.aliases["rededge1"] == [("B05", 1)]
    assert md.aliases["rededge2"] == [("B06", 1)]
    assert md.aliases["rededge3"] == [("B07", 1)]

    # check without config
    md = extract_collection_metadata(item)

    for band in md.bands.values():
        assert band.data_type == "float32"
        assert band.nodata is None
        assert band.unit == "1"

    # Test that multiple CRSs per item work
    item = pystac.Item.from_dict(item0.to_dict())
    ProjectionExtension.ext(item.assets["B01"]).epsg = 3857
    assert ProjectionExtension.ext(item.assets["B01"]).crs_string == "EPSG:3857"
    md = extract_collection_metadata(item, NO_WARN_CFG)
    assert md.band2grid["B01"] != md.band2grid["B02"]

    # Test no-collection name item
    item = pystac.Item.from_dict(item0.to_dict())
    item.collection_id = None
    md = extract_collection_metadata(item, NO_WARN_CFG)
    assert md.name == "_"


def test_parse_item_with_plugin():
    item = pystac.item.Item.from_dict(
        {
            "type": "Feature",
            "id": "some-item",
            "collection": "c",
            "properties": {"datetime": "2022-02-02T00:00:00Z"},
            "geometry": None,
            "links": [],
            "assets": {
                "AA": {
                    "href": "http://example.com/items/b1.nc",
                    "type": "application/x-netcdf",
                    "roles": ["data"],
                }
            },
            "stac_version": "1.0.0",
            "stac_extensions": [""],
        }
    )
    group_md = RasterGroupMetadata(
        bands={
            ("AA", 1): RasterBandMetadata("uint8", 0, "1", dims=("y", "x", "b")),
            ("AA", 2): RasterBandMetadata("float32"),
        },
        aliases={"b1": [("AA", 1)], "b2": [("AA", 2)]},
        extra_dims={"b": 3},
        extra_coords=[FixedCoord("b", ["r", "g", "b"])],
    )
    md_plugin = FakeMDPlugin(group_md, {"foo": "bar"})

    pit = parse_item(item, md_plugin=md_plugin)
    assert isinstance(pit, ParsedItem)
    assert pit.collection["b1"].data_type == "uint8"
    assert pit.collection["b1"].dims == ("y", "x", "b")
    assert pit.collection["b2"].data_type == "float32"
    assert pit["b1"].driver_data == {"foo": "bar"}
    assert pit["b2"].driver_data == {"foo": "bar"}


def test_noassets_case(no_bands_stac):
    md = extract_collection_metadata(no_bands_stac)
    assert len(md.bands) == 0


def test_extract_md_raster_ext(sentinel_stac_ms_with_raster_ext: pystac.item.Item):
    item = sentinel_stac_ms_with_raster_ext

    md = extract_collection_metadata(item, STAC_CFG)

    assert md.aliases["red"] == [("B04", 1), ("visual", 1)]
    assert md.aliases["green"] == [("B03", 1), ("visual", 2)]
    assert md.aliases["blue"] == [("B02", 1), ("visual", 3)]


def test_parse_item(sentinel_stac_ms: pystac.item.Item):
    item0 = sentinel_stac_ms
    item = pystac.Item.from_dict(item0.to_dict())

    md = extract_collection_metadata(item, STAC_CFG)

    xx = parse_item(item, md)
    assert xx.datetime_range == (None, None)
    assert xx.datetime == item.datetime
    assert xx.nominal_datetime == item.datetime

    assert all(band in md for band in S2_ALL_BANDS)
    assert all((band, 1) in md for band in S2_ALL_BANDS)
    assert item not in xx

    assert set(n for n, _ in xx.bands) == S2_ALL_BANDS
    assert xx["B02"].geobox is not None
    assert xx["B02"] is xx[("B02", 1)]
    assert xx["B02"] is xx["B02.1"]
    assert xx.get("B02", None) is xx["B02.1"]

    assert xx.geoboxes() == xx.geoboxes(S2_ALL_BANDS)
    assert xx.geoboxes(["B02", "B03"]) == (xx["B02"].geobox,)
    assert xx.geoboxes(["B01", "B02", "B03"]) == (
        xx["B02"].geobox,
        xx["B01"].geobox,
    )
    assert xx.geoboxes() == (
        xx["B02"].geobox,  # 10m
        xx["B05"].geobox,  # 20m
        xx["B01"].geobox,  # 60m
    )

    (yy,) = list(parse_items(iter([item]), STAC_CFG))
    assert xx == yy

    # Test missing band case
    item = pystac.Item.from_dict(item0.to_dict())
    item.assets.pop("B01")
    xx = parse_item(item, md)
    assert "B01" not in xx
    assert xx.get("B01", None) is None


def test_parse_item_raster_ext(sentinel_stac_ms_with_raster_ext: pystac.item.Item):
    item = sentinel_stac_ms_with_raster_ext
    parsed = parse_item(item)
    assert parsed[("visual", 2)].band == 2
    assert parsed["visual.2"].band == 2
    assert parsed[("visual", 3)] is parsed["visual.3"]

    for (band, idx), b in parsed.bands.items():
        assert idx == b.band
        assert band in S2_ALL_BANDS


@pytest.mark.xfail
def test_parse_no_absolute_href(relative_href_only: pystac.item.Item):
    # Currently pystac never returns `None` from attached asset
    # see: https://github.com/stac-utils/pystac/issues/754
    item = relative_href_only
    assert item.get_self_href() is None
    for asset in item.assets.values():
        assert pystac.utils.is_absolute_href(asset.href) is False
        assert asset.get_absolute_href() is None  # <<< asserts here, but shouldn't

    with pytest.raises(ValueError):
        _ = parse_item(item, extract_collection_metadata(item))


def test_parse_item_no_proj(sentinel_stac_ms: pystac.item.Item):
    item0 = sentinel_stac_ms
    item = pystac.Item.from_dict(item0.to_dict())
    item.stac_extensions.remove(ProjectionExtension.get_schema_uri())
    assert has_proj_ext(item) is False

    md = extract_collection_metadata(item, STAC_CFG)

    xx = parse_item(item, md)
    for band in xx.bands.values():
        assert band.geobox is None

    assert xx.geoboxes() == ()

    assert _auto_load_params([xx] * 3) is None


@pytest.fixture
def parsed_item_s2(sentinel_stac_ms: pystac.item.Item):
    (item,) = parse_items([sentinel_stac_ms], STAC_CFG)
    yield item


def test_auto_load_params(parsed_item_s2: ParsedItem):
    xx = parsed_item_s2
    assert len(xx.geoboxes()) == 3
    crs = xx.geoboxes()[0].crs

    _gbox_10m = xx["B02"].geobox
    _gbox_20m = xx["B05"].geobox
    _gbox_60m = xx["B01"].geobox
    assert _gbox_10m is not None
    assert _gbox_20m is not None
    assert _gbox_60m is not None
    _10m = _gbox_10m.resolution
    _20m = _gbox_20m.resolution
    _60m = _gbox_60m.resolution
    _edge = AnchorEnum.EDGE

    assert _10m.xy == (10, -10)
    assert _20m.xy == (20, -20)
    assert _60m.xy == (60, -60)

    assert _auto_load_params([]) is None
    assert _auto_load_params([xx]) == (crs, _10m, _edge, _gbox_10m)
    assert _auto_load_params([xx] * 3) == (crs, _10m, _edge, _gbox_10m)

    assert _auto_load_params([xx], ["B01"]) == (crs, _60m, _edge, _gbox_60m)
    assert _auto_load_params([xx] * 3, ["B01", "B05", "B06"]) == (
        crs,
        _20m,
        _edge,
        _gbox_20m,
    )
    assert _auto_load_params([xx] * 3, ["B01", "B04"]) == (crs, _10m, _edge, _gbox_10m)


def test_norm_geom(gpd_iso3):
    g = geom.box(0, -1, 10, 1, "epsg:4326")

    assert _normalize_geometry(g) is g
    assert _normalize_geometry(g.geom) == g
    assert _normalize_geometry(g.json) == g

    assert _normalize_geometry(g.geojson()) == g
    assert (
        _normalize_geometry(dict(type="FeatureCollection", features=[g.geojson()])) == g
    )

    g = gpd_iso3("AUS")
    assert g.crs == "epsg:4326"
    assert _normalize_geometry(g).crs == "epsg:4326"

    g = gpd_iso3("AUS", "epsg:3577")
    assert g.crs == "epsg:3577"
    assert _normalize_geometry(g).crs == "epsg:3577"

    with pytest.raises(ValueError):
        _ = _normalize_geometry({})  # not a valid geojson

    with pytest.raises(ValueError):
        _ = _normalize_geometry(object())  # Can't interpret value as geometry


def test_output_geobox(gpd_iso3, parsed_item_s2: ParsedItem):
    au = gpd_iso3("AUS", "epsg:3577")

    gbox = output_geobox([], geopolygon=au, resolution=100, crs="epsg:3857")
    assert gbox is not None
    assert gbox.crs == "epsg:3857"
    assert gbox.resolution.xy == (100, -100)

    # default CRS to that of the polygon
    gbox = output_geobox([], geopolygon=au, resolution=100)
    assert gbox is not None
    assert gbox.crs == "epsg:3577"
    assert gbox.resolution.xy == (100, -100)

    gbox = output_geobox([parsed_item_s2], geopolygon=au, resolution=100)
    assert gbox is not None
    assert gbox.crs == parsed_item_s2.crs()
    assert gbox.resolution.xy == (100, -100)

    gbox = output_geobox([parsed_item_s2])
    assert gbox is not None
    assert gbox.crs == parsed_item_s2.crs()
    assert gbox.resolution.xy == (10, -10)

    # like/gbox
    assert output_geobox([], geobox=gbox) == gbox
    assert output_geobox([], like=gbox) == gbox
    assert output_geobox([], like=xr_zeros(gbox[:10, :20])) == gbox[:10, :20]

    # no resolution/crs
    assert output_geobox([], bbox=(0, 1, 2, 3)) is None
    assert output_geobox([], bbox=(0, 1, 2, 3), resolution=10) is None
    assert output_geobox([], bbox=(0, 1, 2, 3), crs="epsg:4326") is None

    # lon-lat/x-y/bbox
    gbox = GeoBox.from_bbox((0, -10, 100, 25), resolution=1)
    assert gbox.boundingbox == (0, -10, 100, 25)
    bbox = gbox.boundingbox

    assert gbox == output_geobox(
        [],
        x=bbox.range_x,
        y=bbox.range_y,
        resolution=gbox.resolution,
        crs=gbox.crs,
    )

    assert gbox == output_geobox(
        [],
        lon=bbox.range_x,
        lat=bbox.range_y,
        resolution=gbox.resolution,
        crs=gbox.crs,
    )

    assert gbox == output_geobox(
        [],
        bbox=bbox.bbox,
        resolution=gbox.resolution,
        crs=gbox.crs,
    )

    assert gbox == output_geobox(
        [],
        bbox=bbox.bbox,
        resolution=gbox.resolution,
        crs=gbox.crs,
        align=0,
    )


def test_output_geobox_from_items():
    cc = 0

    def mk_item(gbox: GeoBox, time="2020-01-10"):
        nonlocal cc
        cc = cc + 1
        return mk_parsed_item(
            [b_("b1", geobox=gbox), b_("b2", geobox=gbox)], time, id=f"item-{cc}"
        )

    gboxes = [GBOX, GBOX.left, GBOX.right.pad(3)]

    gbox = output_geobox([mk_item(gbox) for gbox in gboxes])
    assert gbox.crs == GBOX.crs
    assert geobox_union_conservative(gboxes) == gbox


@pytest.mark.parametrize(
    "kw",
    [
        # not enough args
        {"x": (0, 10)},
        {"y": (0, 10)},
        {"lon": (0, 1)},
        {"lat": (0, 1)},
        {"x": (0, 1), "y": (1, 2)},
        # too many args
        {"lat": (0, 1), "lon": (1, 2), "x": (3, 4), "y": (5, 6)},
        {"lat": (0, 1), "lon": (1, 2), "bbox": (0, 1, 2, 3)},
        {"lat": (0, 1), "lon": (1, 2), "geopolygon": geom.box(0, 0, 1, 1, "epsg:4326")},
        {"bbox": (0, 0, 1, 1), "geopolygon": geom.box(0, 0, 1, 1, "epsg:4326")},
        # bad args
        {"like": object()},
    ],
)
def test_output_gbox_bads(kw):
    with pytest.raises(ValueError):
        _ = output_geobox([], **kw)


def test_mk_parsed_item():
    fmt = "%Y-%m-%d"
    item = mk_parsed_item(
        [b_("b1"), b_("b2")],
        "2020-01-10",
        start_datetime="2020-01-01",
        end_datetime="2020-01-31",
    )

    assert item.datetime.strftime(fmt) == "2020-01-10"
    assert item.datetime_range[0].strftime(fmt) == "2020-01-01"
    assert item.datetime_range[1].strftime(fmt) == "2020-01-31"
    assert item.geometry is None
    assert item.crs() is None
    assert item.collection.has_proj is False

    assert set(item.bands) == set([("b1", 1), ("b2", 1)])
    assert item["b1"].uri.endswith("b1.tif")

    item = mk_parsed_item(
        [b_("b1"), b_("b2")],
        datetime=None,
        start_datetime="2020-01-01",
        end_datetime="2020-01-31",
    )

    assert item.datetime is None
    assert item.datetime_range[0].strftime(fmt) == "2020-01-01"
    assert item.datetime_range[1].strftime(fmt) == "2020-01-31"

    item = mk_parsed_item(
        [b_("b1"), b_("b2")],
        "2020-01-10",
        start_datetime="2020-01-01",
        end_datetime=None,
    )
    assert item.datetime.strftime(fmt) == "2020-01-10"
    assert item.datetime_range[0].strftime(fmt) == "2020-01-01"
    assert item.datetime_range[1] is None

    gbox = GeoBox.from_bbox((-20, -10, 20, 10), "epsg:3857", shape=(200, 400))
    item = mk_parsed_item(
        [b_("b1", geobox=gbox), b_("b2", geobox=gbox)],
        "2020-01-10",
    )
    assert item.geometry is not None
    assert item.geometry.crs == "epsg:4326"
    assert item.crs() == "epsg:3857"
    assert item.geoboxes() == (gbox,)
    assert item.collection.has_proj is True


@pytest.mark.parametrize(
    "parsed_item",
    [
        mk_parsed_item(
            [b_("band")], None, "2020-01-01", "2021-12-31T23:59:59.9999999Z"
        ),
        mk_parsed_item([b_("b1"), b_("b2", nodata=10)], "2020-01-01"),
        mk_parsed_item(
            [
                b_("b1", dtype="float32", geobox=GBOX),
                b_("b2", nodata=10, geobox=GBOX),
            ],
            "2020-01-01",
        ),
        mk_parsed_item(
            [
                b_("b1", dtype="float32", geobox=GBOX),
                b_("b2", dtype="int32", nodata=-99, geobox=GBOX.zoom_out(2)),
            ],
            "2020-01-01",
            "2020-01-01",
            "2021-12-31T23:59:59.9999999Z",
            href="file:///date/item/1.json",
        ),
    ],
)
def test_round_trip(parsed_item: ParsedItem):
    item = to_stac_item(parsed_item)
    md = extract_collection_metadata(item)

    assert parsed_item.collection == md
    assert parsed_item == parse_item(item, md)


def test_usgs_v1_1_1_aliases(usgs_landsat_stac_v1_1_1: pystac.Item) -> None:
    parsed_item = next(parse_items([usgs_landsat_stac_v1_1_1]))
    collection = parsed_item.collection
    assert collection.aliases == {
        "B1": [("blue", 1)],
        "B2": [("green", 1)],
        "B3": [("red", 1)],
        "B4": [("nir08", 1)],
        "B5": [("swir16", 1)],
        "B7": [("swir22", 1)],
    }


@pytest.mark.parametrize(
    "gbox",
    [
        GeoBox.from_bbox((0, 0, 100, 200), resolution=10, crs=3857),
        GeoBox.from_bbox((-10, 0, 100, 200), resolution=10, crs=3857),
    ],
)
def test_gbox_anchor(gbox: GeoBox):
    assert _gbox_anchor(gbox) == AnchorEnum.EDGE
    assert _gbox_anchor(gbox.translate_pix(-1e-5, 1e-5)) == AnchorEnum.EDGE
    assert _gbox_anchor(gbox.translate_pix(0.5, 0.5)) == AnchorEnum.CENTER
    assert _gbox_anchor(gbox.translate_pix(-1 / 4, -1 / 8)) == xy_(1 / 4, 1 / 8)


def test_most_common_gbox():
    gbox = GeoBox.from_bbox((0, 0, 100, 200), resolution=10, crs=3857)
    assert _most_common_gbox(
        [gbox, gbox.center_pixel, gbox[:1, :1], gbox.zoom_out(1.3)]
    ) == (
        gbox.crs,
        gbox.resolution,
        AnchorEnum.EDGE,
        None,
    )
    # not enough consensus for anchor
    # fallback to EDGE aligned
    assert _most_common_gbox(
        [
            gbox.translate_pix(-0.3, -0.2),
            gbox.center_pixel.translate_pix(-0.1, -0.4),
            gbox.zoom_out(1.3),
            gbox.to_crs(4326),
        ],
        1 / 4 + 0.1,
    ) == (
        gbox.crs,
        gbox.resolution,
        AnchorEnum.EDGE,
        None,
    )

    # CENTER
    gbox = GeoBox.from_bbox(
        (0, 0, 100, 200), resolution=10, crs=3857, anchor=AnchorEnum.CENTER
    )
    assert _most_common_gbox(
        [gbox, gbox.center_pixel, gbox[:1, :1], gbox.zoom_out(1.3)]
    ) == (
        gbox.crs,
        gbox.resolution,
        AnchorEnum.CENTER,
        None,
    )
