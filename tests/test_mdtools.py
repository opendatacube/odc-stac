import math

import pystac
import pystac.asset
import pystac.collection
import pystac.item
import pystac.utils
import pytest
from common import NO_WARN_CFG, S2_ALL_BANDS, STAC_CFG
from pystac.extensions.projection import ProjectionExtension

from odc.stac._mdtools import (
    RasterBandMetadata,
    asset_geobox,
    band_metadata,
    compute_eo3_grids,
    extract_collection_metadata,
    has_proj_ext,
    is_raster_data,
    parse_item,
)


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
    ProjectionExtension.ext(item.assets["B01"]).shape = (100, 200)
    with pytest.raises(NotImplementedError):
        compute_eo3_grids(data_bands)

    # More than 1 CRS is not supported
    item = item0.clone()
    ProjectionExtension.ext(item.assets["B01"]).epsg = 3857
    with pytest.raises(ValueError):
        compute_eo3_grids(data_bands)


def test_asset_geobox(sentinel_stac: pystac.item.Item):
    item0 = sentinel_stac
    item = item0.clone()
    asset = item.assets["B01"]
    geobox = asset_geobox(asset)
    assert geobox.shape == (1830, 1830)

    # Tests non-affine transofrm ValueError
    item = item0.clone()
    asset = item.assets["B01"]
    ProjectionExtension.ext(asset).transform[-1] = 2
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
    asset = item.assets["SCL"]
    bm = band_metadata(asset, RasterBandMetadata("uint16", 0, "1"))
    assert bm == RasterBandMetadata("uint8", 0, "1")

    # Test multiple bands per asset cause a warning
    asset.extra_fields["raster:bands"].append({"nodata": -10})
    with pytest.warns(UserWarning, match="Defaulting to first band of 2"):
        bm = band_metadata(asset, RasterBandMetadata("uint16", 0, "1"))
    assert bm == RasterBandMetadata("uint8", 0, "1")


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

    with pytest.warns(UserWarning, match="Common name `rededge` is repeated, skipping"):
        md = extract_collection_metadata(item, STAC_CFG)

    assert md.name == "sentinel-2-l2a"

    assert set(md.bands) == S2_ALL_BANDS

    # check defaults were set
    for b in ["B01", "B02", "B03"]:
        assert md.bands[b].data_type == "uint16"
        assert md.bands[b].nodata == 0
        assert md.bands[b].unit == "1"

    assert md.bands["SCL"].data_type == "uint8"
    assert md.bands["visual"].data_type == "uint8"

    # check aliases configuration
    assert md.aliases["rededge"] == "B05"
    assert md.aliases["rededge1"] == "B05"
    assert md.aliases["rededge2"] == "B06"
    assert md.aliases["rededge3"] == "B07"

    # check without config
    with pytest.warns(UserWarning, match="Common name `rededge` is repeated, skipping"):
        md = extract_collection_metadata(item)

    for band in md.bands.values():
        assert band.data_type == "float32"
        assert math.isnan(band.nodata)
        assert band.unit == "1"

    # Test multiple CRS unhappy path
    item = pystac.Item.from_dict(item0.to_dict())
    ProjectionExtension.ext(item.assets["B01"]).epsg = 3857
    assert ProjectionExtension.ext(item.assets["B01"]).crs_string == "EPSG:3857"
    with pytest.raises(ValueError):
        with pytest.warns(UserWarning, match="`rededge`"):
            md = extract_collection_metadata(item, STAC_CFG)

    # Test no-collection name item
    item = pystac.Item.from_dict(item0.to_dict())
    item.collection_id = None
    md = extract_collection_metadata(item, NO_WARN_CFG)
    assert md.name == "_"


def test_noassets_case(no_bands_stac):
    with pytest.raises(ValueError):
        _ = extract_collection_metadata(no_bands_stac)


def test_extract_md_raster_ext(sentinel_stac_ms_with_raster_ext: pystac.item.Item):
    item = sentinel_stac_ms_with_raster_ext

    with pytest.warns(UserWarning, match="Common name `rededge` is repeated, skipping"):
        md = extract_collection_metadata(item, STAC_CFG)

    assert md.aliases["red"] == "B04"
    assert md.aliases["green"] == "B03"
    assert md.aliases["blue"] == "B02"


def test_parse_item(sentinel_stac_ms: pystac.item.Item):
    item0 = sentinel_stac_ms
    item = pystac.Item.from_dict(item0.to_dict())

    with pytest.warns(UserWarning, match="Common name `rededge` is repeated, skipping"):
        md = extract_collection_metadata(item, STAC_CFG)

    xx = parse_item(item, md)

    assert set(xx.bands) == S2_ALL_BANDS

    assert xx.bands["B02"].geobox is not None

    # Test missing band case
    item = pystac.Item.from_dict(item0.to_dict())
    item.assets.pop("B01")
    with pytest.warns(UserWarning, match="Missing asset"):
        xx = parse_item(item, md)


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
    item.stac_extensions.remove(
        "https://stac-extensions.github.io/projection/v1.0.0/schema.json"
    )
    assert has_proj_ext(item) is False

    with pytest.warns(UserWarning, match="`rededge`"):
        md = extract_collection_metadata(item, STAC_CFG)

    xx = parse_item(item, md)
    for band in xx.bands.values():
        assert band.geobox is None
