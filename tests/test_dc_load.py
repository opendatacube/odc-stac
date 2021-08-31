import pytest
import pystac

from odc.stac import dc_load, stac2ds


def test_dc_load_smoketest(sentinel_stac_ms):
    item = pystac.Item.from_dict(sentinel_stac_ms)
    with pytest.warns(UserWarning, match="`rededge`"):
        (ds,) = stac2ds([item], {})

    params = dict(output_crs=ds.crs, resolution=(-100, 100), chunks={})
    xx = dc_load([ds], "B02", **params)
    assert xx.B02.shape == (1, 1099, 1099)
    assert xx.B02.geobox.crs == ds.crs

    # Check that aliases also work
    xx = dc_load([ds], ["red", "green", "blue"], **params)
    assert xx.green.shape == xx.red.shape
    assert xx.blue.dtype == xx.red.dtype
