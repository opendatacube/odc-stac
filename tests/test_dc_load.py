import pystac

from odc.stac import dc_load, stac2ds


def test_dc_load_smoketest(sentinel_stac_ms):
    item = pystac.Item.from_dict(sentinel_stac_ms)
    (ds,) = stac2ds([item], {})
    xx = dc_load([ds], "B02", output_crs=ds.crs, resolution=(-100, 100), chunks={})
    assert xx.B02.shape == (1, 1099, 1099)
    assert xx.B02.geobox.crs == ds.crs
