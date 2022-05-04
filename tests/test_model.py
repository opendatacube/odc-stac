import datetime as dt

import pytest
from odc.geo.geobox import GeoBox

from odc.stac._model import RasterBandMetadata, RasterLoadParams, RasterSource
from odc.stac.testing.stac import b_, mk_parsed_item


def test_band_load_info():
    meta = RasterBandMetadata(data_type="uint16", nodata=13)
    band = RasterSource("https://example.com/some.tif", meta=meta)
    assert RasterLoadParams.same_as(meta).dtype == "uint16"
    assert RasterLoadParams.same_as(band).fill_value == 13

    band = RasterSource("file:///")
    assert RasterLoadParams.same_as(band).dtype == "float32"
    assert RasterLoadParams().dtype == None
    assert RasterLoadParams().nearest is True
    assert RasterLoadParams(resampling="average").nearest is False


@pytest.mark.parametrize("lon", [0, -179, 179, 10, 23.4])
def test_mid_longitude(lon: float):
    gbox = GeoBox.from_bbox([lon - 0.1, 0, lon + 0.1, 1], shape=(100, 100))
    xx = mk_parsed_item([b_("b1", gbox)])
    assert xx.geometry is not None
    assert xx.geometry.crs == "epsg:4326"
    assert xx.mid_longitude == pytest.approx(lon)

    assert mk_parsed_item([]).mid_longitude is None


def test_solar_day():
    def _mk(lon: float, datetime):
        gbox = GeoBox.from_bbox([lon - 0.1, 0, lon + 0.1, 1], shape=(100, 100))
        return mk_parsed_item([b_("b1", gbox)], datetime=datetime)

    for lon in [0, 1, 2, 3, 14, -1, -14, -3]:
        xx = _mk(lon, "2020-01-02T12:13:14Z")
        assert xx.mid_longitude == pytest.approx(lon)
        assert xx.nominal_datetime == xx.solar_date

    xx = _mk(15.1, "2020-01-02T12:13:14Z")
    assert xx.nominal_datetime != xx.solar_date
    assert xx.nominal_datetime + dt.timedelta(seconds=3600) == xx.solar_date

    xx = _mk(-15.1, "2020-01-02T12:13:14Z")
    assert xx.nominal_datetime != xx.solar_date
    assert xx.nominal_datetime - dt.timedelta(seconds=3600) == xx.solar_date

    xx = mk_parsed_item([b_("b1")], datetime="2000-01-02")
    assert xx.geometry is None
    assert xx.nominal_datetime == xx.solar_date

    xx = _mk(10, None)
    with pytest.raises(ValueError):
        _ = xx.solar_date
