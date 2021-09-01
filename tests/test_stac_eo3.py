import pytest
from odc.stac._eo3 import (
    mk_product,
    BandMetadata,
    compute_eo3_grids,
    infer_dc_product,
    is_raster_data,
    item_to_ds,
    stac2ds,
    asset_geobox,
    has_proj_ext,
)
import pystac
from pystac.extensions.projection import ProjectionExtension

STAC_CFG = {
    "sentinel-2-l2a": {
        "measurements": {
            "*": BandMetadata("uint16", 0, "1"),
            "SCL": BandMetadata("uint8", 0, "1"),
            "visual": dict(dtype="uint8", nodata=0, units="1"),
        },
        "aliases": {  # Work around duplicate rededge common_name
            "rededge": "B05",
            "rededge1": "B05",
            "rededge2": "B06",
            "rededge3": "B07",
        },
    }
}


def test_mk_product():
    p = mk_product(
        "some-product",
        ["a", "b"],
        {"*": BandMetadata("uint8", 0, "1"), "b": BandMetadata("int16", -999, "BB")},
        {"A": "a", "B": "b", "bb": "b"},
    )

    assert p.name == "some_product"
    assert p.metadata_type.name == "eo3"
    assert set(p.measurements) == set(["a", "b"])
    assert p.measurements["a"].dtype == "uint8"
    assert p.measurements["a"].nodata == 0
    assert p.measurements["a"].units == "1"
    assert p.measurements["a"].aliases == ["A"]

    assert p.measurements["b"].dtype == "int16"
    assert p.measurements["b"].nodata == -999
    assert p.measurements["b"].units == "BB"
    assert p.canonical_measurement("B") == "b"
    assert p.canonical_measurement("bb") == "b"

    p = mk_product("Some Product", ["a", "b", "c"], {},)

    assert p.name == "Some_Product"
    assert set(p.measurements) == set(["a", "b", "c"])
    assert p.metadata_type.name == "eo3"

    for m in p.measurements.values():
        assert m.dtype == "uint16"
        assert m.nodata == 0
        assert m.units == "1"


def test_is_raster_data(sentinel_stac_ms):
    item = pystac.Item.from_dict(sentinel_stac_ms)
    assert "B01" in item.assets
    assert "B02" in item.assets

    assert is_raster_data(item.assets["B01"])

    # check case when roles are missing
    item.assets["B02"].roles = None
    assert is_raster_data(item.assets["B02"])


def test_eo3_grids(sentinel_stac_ms):
    item0 = pystac.Item.from_dict(sentinel_stac_ms)

    item = item0.full_copy()
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
    item = item0.full_copy()
    ProjectionExtension.ext(item.assets["B01"]).epsg = 3857
    with pytest.raises(ValueError):
        compute_eo3_grids(data_bands)


def test_infer_product(sentinel_stac_ms):
    item = pystac.Item.from_dict(sentinel_stac_ms)

    assert item.collection_id in STAC_CFG

    with pytest.warns(UserWarning, match="Common name `rededge` is repeated, skipping"):
        product = infer_dc_product(item, STAC_CFG)

    assert product.measurements["SCL"].dtype == "uint8"
    assert product.measurements["visual"].dtype == "uint8"
    # check aliases from eo extension
    assert product.canonical_measurement("red") == "B04"
    assert product.canonical_measurement("green") == "B03"
    assert product.canonical_measurement("blue") == "B02"
    # check aliases from config
    assert product.canonical_measurement("rededge") == "B05"
    assert product.canonical_measurement("rededge1") == "B05"
    assert product.canonical_measurement("rededge2") == "B06"
    assert product.canonical_measurement("rededge3") == "B07"

    assert set(product._stac_cfg["band2grid"]) == set(product.measurements)


def test_item_to_ds(sentinel_stac_ms):
    item0 = pystac.Item.from_dict(sentinel_stac_ms)
    item = item0.full_copy()

    assert item.collection_id in STAC_CFG

    with pytest.warns(UserWarning, match="`rededge`"):
        product = infer_dc_product(item, STAC_CFG)
    ds = item_to_ds(item, product)

    assert set(ds.measurements) == set(product.measurements)
    assert ds.crs is not None
    assert ds.metadata.lat is not None
    assert ds.metadata.lon is not None
    assert ds.center_time is not None

    with pytest.warns(UserWarning, match="`rededge`"):
        dss = list(stac2ds(iter([item, item, item]), STAC_CFG))
    assert len(dss) == 3
    assert len(set(id(ds.type) for ds in dss)) == 1

    # Test missing band case
    item = item0.full_copy()
    item.assets.pop("B01")
    with pytest.warns(UserWarning, match="Missing asset"):
        ds = item_to_ds(item, product)

    # Test no eo extension case
    item = item0.full_copy()
    item.stac_extensions.remove(
        "https://stac-extensions.github.io/eo/v1.0.0/schema.json"
    )
    product = infer_dc_product(item, STAC_CFG)
    with pytest.raises(ValueError):
        product.canonical_measurement("green")

    # Test multiple CRS unhappy path
    item = item0.full_copy()
    ProjectionExtension.ext(item.assets["B01"]).epsg = 3857
    assert ProjectionExtension.ext(item.assets["B01"]).crs_string == "EPSG:3857"
    with pytest.raises(ValueError):
        with pytest.warns(UserWarning, match="`rededge`"):
            infer_dc_product(item, STAC_CFG)


def test_asset_geobox(sentinel_stac):
    item0 = pystac.Item.from_dict(sentinel_stac)

    item = item0.full_copy()
    asset = item.assets["B01"]
    geobox = asset_geobox(asset)
    assert geobox.shape == (1830, 1830)

    # Tests non-affine transofrm ValueError
    item = item0.full_copy()
    asset = item.assets["B01"]
    ProjectionExtension.ext(asset).transform[-1] = 2
    with pytest.raises(ValueError):
        asset_geobox(asset)

    # Tests wrong-sized transform transofrm ValueError
    item = item0.full_copy()
    asset = item.assets["B01"]
    ProjectionExtension.ext(asset).transform = [1, 1, 2]
    with pytest.raises(ValueError):
        asset_geobox(asset)

    # Test missing transform transofrm ValueError
    item = item0.full_copy()
    asset = item.assets["B01"]
    ProjectionExtension.ext(asset).transform = None
    with pytest.raises(ValueError):
        asset_geobox(asset)


def test_has_proj_ext(sentinel_stac_ms_no_ext):
    item = pystac.Item.from_dict(sentinel_stac_ms_no_ext)
    assert has_proj_ext(item) is False
