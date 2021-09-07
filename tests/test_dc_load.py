from mock import MagicMock
import pytest
import pystac

from odc.stac import dc_load, stac2ds, stac_load


def test_dc_load_smoketest(sentinel_stac_ms):
    item = pystac.Item.from_dict(sentinel_stac_ms)
    with pytest.warns(UserWarning, match="`rededge`"):
        (ds,) = stac2ds([item], {})

    params = dict(output_crs=ds.crs, resolution=(-100, 100), chunks={})
    xx = dc_load([ds], "B02", **params)
    assert xx.B02.shape == (1, 1099, 1099)
    assert xx.B02.geobox.crs == ds.crs

    # Check that aliases also work
    xx = dc_load([ds], bands=["red", "green", "blue"], **params)
    assert xx.green.shape == xx.red.shape
    assert xx.blue.dtype == xx.red.dtype

    # Check that dask_chunks= is an alias for chunks=
    params["dask_chunks"] = params.pop("chunks")
    xx = dc_load([ds], "nir", **params)
    assert xx.nir.chunks == tuple((s,) for s in xx.nir.shape)

    with pytest.warns(UserWarning, match="Supplied 'geobox=' parameter aliases"):
        yy = dc_load([ds], "SCL", geobox=xx.nir.geobox, **params)
    assert xx.nir.geobox == yy.SCL.geobox


def test_stac_load_smoketest(sentinel_stac_ms_with_raster_ext: pystac.Item):
    item = sentinel_stac_ms_with_raster_ext.clone()

    params = dict(output_crs="EPSG:3857", resolution=(-100, 100), chunks={})
    with pytest.warns(UserWarning, match="`rededge`"):
        xx = stac_load([item], "B02", **params)

    assert xx.B02.shape[0] == 1
    assert xx.B02.geobox.crs == "EPSG:3857"

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
        **params
    )
    # expect patch_url to be called 2 times, 1 for red and 1 for green band
    assert patch_url.call_count == 2
