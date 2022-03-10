from odc.stac._model import RasterBandMetadata, RasterLoadParams, RasterSource


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
