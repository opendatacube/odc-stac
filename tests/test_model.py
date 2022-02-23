from odc.stac._model import RasterBandMetadata, RasterLoadParams, RasterSource


def test_band_load_info():
    meta = RasterBandMetadata(data_type="uint16", nodata=13)
    band = RasterSource("https://example.com/some.tif", meta=meta)
    assert RasterLoadParams(band).dtype == "uint16"
    assert RasterLoadParams(band).fill_value == 13

    band = RasterSource("file:///")
    assert RasterLoadParams(band, dtype="uint16").dtype == "uint16"
    assert RasterLoadParams(band).dtype is None
