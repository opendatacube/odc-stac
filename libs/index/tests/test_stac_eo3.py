from odc.stac._eo3 import (
    mk_product,
    BandMetadata,
    compute_eo3_grids,
    infer_dc_product,
    is_raster_data,
    item_to_ds,
    stac2ds,
)
import pystac

STAC_CFG = {
    "sentinel-2-l2a": {
        "measurements": {
            "*": BandMetadata("uint16", 0, "1"),
            "SCL": BandMetadata("uint8", 0, "1"),
            "visual": dict(dtype="uint8", nodata=0, units="1"),
        }
    }
}


def test_mk_product():
    p = mk_product(
        "some-product",
        ["a", "b"],
        {"*": BandMetadata("uint8", 0, "1"), "b": BandMetadata("int16", -999, "BB")},
    )

    assert p.name == "some_product"
    assert p.metadata_type.name == "eo3"
    assert set(p.measurements) == set(["a", "b"])
    assert p.measurements["a"].dtype == "uint8"
    assert p.measurements["a"].nodata == 0
    assert p.measurements["a"].units == "1"

    assert p.measurements["b"].dtype == "int16"
    assert p.measurements["b"].nodata == -999
    assert p.measurements["b"].units == "BB"

    p = mk_product("Some Product", ["a", "b", "c"], {},)

    assert p.name == "Some_Product"
    assert set(p.measurements) == set(["a", "b", "c"])
    assert p.metadata_type.name == "eo3"

    for m in p.measurements.values():
        assert m.dtype == "uint16"
        assert m.nodata == 0
        assert m.units == "1"


def test_eo3_grids(sentinel_stac_ms):
    item = pystac.Item.from_dict(sentinel_stac_ms)
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


def test_infer_product(sentinel_stac_ms):
    item = pystac.Item.from_dict(sentinel_stac_ms)

    assert item.collection_id in STAC_CFG

    product = infer_dc_product(item, STAC_CFG)
    assert product.measurements["SCL"].dtype == "uint8"
    assert product.measurements["visual"].dtype == "uint8"

    assert set(product._stac_cfg["band2grid"]) == set(product.measurements)


def test_item_to_ds(sentinel_stac_ms):
    item = pystac.Item.from_dict(sentinel_stac_ms)

    assert item.collection_id in STAC_CFG

    product = infer_dc_product(item, STAC_CFG)
    ds = item_to_ds(item, product)

    assert set(ds.measurements) == set(product.measurements)
    assert ds.crs is not None
    assert ds.metadata.lat is not None
    assert ds.metadata.lon is not None
    assert ds.center_time is not None

    dss = list(stac2ds(iter([item, item, item]), STAC_CFG))
    assert len(dss) == 3
    assert len(set(id(ds.type) for ds in dss)) == 1
