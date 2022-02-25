import uuid

import pystac
import pystac.asset
import pystac.collection
import pystac.item
import pytest
from common import mk_stac_item
from datacube.testutils.io import native_geobox
from datacube.utils.geometry import Geometry
from pystac.extensions.projection import ProjectionExtension
from toolz import dicttoolz

from odc.stac._eo3converter import (
    _compute_uuid,
    infer_dc_product,
    item_to_ds,
    mk_product,
    stac2ds,
)
from odc.stac._mdtools import (
    RasterBandMetadata,
    asset_geobox,
    band_metadata,
    compute_eo3_grids,
    has_proj_ext,
    is_raster_data,
)

STAC_CFG = {
    "sentinel-2-l2a": {
        "assets": {
            "*": RasterBandMetadata("uint16", 0, "1"),
            "SCL": RasterBandMetadata("uint8", 0, "1"),
            "visual": dict(data_type="uint8", nodata=0, unit="1"),
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
        {
            "*": RasterBandMetadata("uint8", 0, "1"),
            "b": RasterBandMetadata("int16", -999, "BB"),
        },
        {"A": "a", "B": "b", "bb": "b"},
    )

    assert p.name == "some_product"
    assert p.metadata_type.name == "eo3"
    assert set(p.measurements) == {"a", "b"}
    assert p.measurements["a"].dtype == "uint8"
    assert p.measurements["a"].nodata == 0
    assert p.measurements["a"].units == "1"
    assert p.measurements["a"].aliases == ["A"]

    assert p.measurements["b"].dtype == "int16"
    assert p.measurements["b"].nodata == -999
    assert p.measurements["b"].units == "BB"
    assert p.canonical_measurement("B") == "b"
    assert p.canonical_measurement("bb") == "b"

    p = mk_product(
        "Some Product",
        ["a", "b", "c"],
        {},
    )

    assert p.name == "Some_Product"
    assert set(p.measurements) == {"a", "b", "c"}
    assert p.metadata_type.name == "eo3"

    for m in p.measurements.values():
        assert m.dtype == "uint16"
        assert m.nodata == 0
        assert m.units == "1"


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


def test_infer_product_collection(
    sentinel_stac_collection: pystac.collection.Collection,
    sentinel_stac_ms_with_raster_ext: pystac.item.Item,
):

    with pytest.warns(UserWarning):
        product = infer_dc_product(sentinel_stac_collection)
    assert product.measurements["SCL"].dtype == "uint8"
    # check aliases from eo extension
    assert product.canonical_measurement("red") == "B04"
    assert product.canonical_measurement("green") == "B03"
    assert product.canonical_measurement("blue") == "B02"

    # check band2grid
    b2g = product._stac_cfg["band2grid"]
    assert b2g["B02"] == "default"
    assert b2g["B01"] == "g60"
    assert set(b2g.values()) == set("default g20 g60".split(" "))
    assert set(b2g) == set(product.measurements)

    # Check that we can use product derived this way on an Item
    item = sentinel_stac_ms_with_raster_ext.clone()
    ds = item_to_ds(item, product)
    geobox = native_geobox(ds, basis="B02")
    assert geobox.shape == (10980, 10980)
    assert geobox.crs == "EPSG:32606"
    assert native_geobox(ds, basis="B01").shape == (1830, 1830)

    # Check unhappy path
    collection = sentinel_stac_collection.clone()
    collection.stac_extensions.remove(
        "https://stac-extensions.github.io/item-assets/v1.0.0/schema.json"
    )
    with pytest.raises(ValueError):
        infer_dc_product(collection)

    # Test bad overload
    with pytest.raises(TypeError):
        infer_dc_product([])


def test_infer_product_item(sentinel_stac_ms: pystac.item.Item):
    item = sentinel_stac_ms

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

    _stac = dicttoolz.dissoc(sentinel_stac_ms.to_dict(), "collection")
    item_no_collection = pystac.item.Item.from_dict(_stac)
    assert item_no_collection.collection_id is None

    with pytest.warns(UserWarning, match="Common name `rededge` is repeated, skipping"):
        product = infer_dc_product(item_no_collection)


def test_infer_product_raster_ext(sentinel_stac_ms_with_raster_ext: pystac.item.Item):
    item = sentinel_stac_ms_with_raster_ext.clone()
    with pytest.warns(UserWarning, match="Common name `rededge` is repeated, skipping"):
        product = infer_dc_product(item)

    assert product.measurements["SCL"].dtype == "uint8"
    assert product.measurements["visual"].dtype == "uint8"

    # check aliases from eo extension
    assert product.canonical_measurement("red") == "B04"
    assert product.canonical_measurement("green") == "B03"
    assert product.canonical_measurement("blue") == "B02"
    assert set(product._stac_cfg["band2grid"]) == set(product.measurements)


def test_item_to_ds(sentinel_stac_ms: pystac.item.Item):
    item0 = sentinel_stac_ms
    item = item0.clone()

    assert item.collection_id in STAC_CFG

    with pytest.warns(UserWarning, match="`rededge`"):
        product = infer_dc_product(item, STAC_CFG)
    ds = item_to_ds(item, product)

    assert set(ds.measurements) == set(product.measurements)
    assert ds.crs is not None
    assert ds.metadata.lat is not None
    assert ds.metadata.lon is not None
    assert ds.center_time is not None

    # this checks property remap, without changing
    # key names .platform would be None
    assert ds.metadata.platform == "Sentinel-2B"

    with pytest.warns(UserWarning, match="`rededge`"):
        dss = list(stac2ds(iter([item, item, item]), STAC_CFG))
    assert len(dss) == 3
    assert len({id(ds.type) for ds in dss}) == 1

    # Test missing band case
    item = item0.clone()
    item.assets.pop("B01")
    with pytest.warns(UserWarning, match="Missing asset"):
        ds = item_to_ds(item, product)

    # Test no eo extension case
    item = item0.clone()
    item.stac_extensions.remove(
        "https://stac-extensions.github.io/eo/v1.0.0/schema.json"
    )
    product = infer_dc_product(item, STAC_CFG)
    with pytest.raises(ValueError):
        product.canonical_measurement("green")

    # Test multiple CRS unhappy path
    item = item0.clone()
    ProjectionExtension.ext(item.assets["B01"]).epsg = 3857
    assert ProjectionExtension.ext(item.assets["B01"]).crs_string == "EPSG:3857"
    with pytest.raises(ValueError):
        with pytest.warns(UserWarning, match="`rededge`"):
            infer_dc_product(item, STAC_CFG)


def test_item_to_ds_no_proj(sentinel_stac_ms: pystac.item.Item):
    item0 = sentinel_stac_ms
    item = item0.clone()
    item.stac_extensions.remove(
        "https://stac-extensions.github.io/projection/v1.0.0/schema.json"
    )
    assert has_proj_ext(item) is False

    with pytest.warns(UserWarning, match="`rededge`"):
        product = infer_dc_product(item, STAC_CFG)

    geom = Geometry(item.geometry, "EPSG:4326")
    ds = item_to_ds(item, product)
    assert ds.crs == "EPSG:4326"
    assert ds.extent is not None
    assert ds.extent.contains(geom)
    assert native_geobox(ds).shape == (1, 1)


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


def test_item_uuid():
    item1 = mk_stac_item("id1", custom_property=1)
    item2 = mk_stac_item("id2")

    # Check determinism
    assert _compute_uuid(item1) == _compute_uuid(item1)
    assert _compute_uuid(item2) == _compute_uuid(item2)
    assert _compute_uuid(item1) != _compute_uuid(item2)

    # Check random case
    assert _compute_uuid(item1, "random").version == 4
    assert _compute_uuid(item1, "random") != _compute_uuid(item1, "random")

    # Check "native" mode
    _id = uuid.uuid4()
    assert _compute_uuid(mk_stac_item(str(_id)), "native") == _id
    assert _compute_uuid(mk_stac_item(str(_id)), "auto") == _id

    # Check that extras are used
    id1 = _compute_uuid(item1, extras=["custom_property", "missing_property"])
    id2 = _compute_uuid(item1)

    assert id1.version == 5
    assert id2.version == 5
    assert id1 != id2


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


def test_issue_n6(usgs_landsat_stac_v1):
    expected_bands = {
        "blue",
        "coastal",
        "green",
        "nir08",
        "red",
        "swir16",
        "swir22",
        "qa_aerosol",
        "qa_pixel",
        "qa_radsat",
    }
    p = infer_dc_product(usgs_landsat_stac_v1)
    assert set(p.measurements) == expected_bands


def test_partial_proj(partial_proj_stac):
    (ds,) = list(stac2ds([partial_proj_stac]))
    assert ds.metadata_doc["grids"]["default"]["shape"] == (1, 1)


def test_noassets_case(no_bands_stac):
    with pytest.raises(ValueError):
        list(stac2ds([no_bands_stac]))
