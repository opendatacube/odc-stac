from math import isnan

import numpy as np
import pytest
import rasterio
from numpy import ma
from numpy.testing import assert_array_equal
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_zeros

from odc.stac._model import RasterLoadParams, RasterSource
from odc.stac._reader import (
    _pick_overview,
    _resolve_dst_dtype,
    _resolve_dst_nodata,
    _resolve_src_nodata,
    _same_nodata,
    rio_read,
)
from odc.stac.testing.fixtures import with_temp_tiff


def test_same_nodata():
    _nan = float("nan")
    assert _same_nodata(None, None) is True
    assert _same_nodata(_nan, _nan) is True
    assert _same_nodata(1, None) is False
    assert _same_nodata(_nan, None) is False
    assert _same_nodata(None, _nan) is False
    assert _same_nodata(10, _nan) is False
    assert _same_nodata(_nan, 10) is False
    assert _same_nodata(109, 109) is True
    assert _same_nodata(109, 1) is False


def test_resolve_nodata():
    def _cfg(**kw):
        return RasterLoadParams("uint8", **kw)

    assert _resolve_src_nodata(None, _cfg()) is None
    assert _resolve_src_nodata(11, _cfg()) == 11
    assert _resolve_src_nodata(None, _cfg(src_nodata_fallback=0)) == 0
    assert _resolve_src_nodata(None, _cfg(src_nodata_fallback=11)) == 11
    assert _resolve_src_nodata(11, _cfg(src_nodata_fallback=0)) == 11
    assert _resolve_src_nodata(11, _cfg(src_nodata_override=-1)) == -1
    assert _resolve_src_nodata(11, _cfg(src_nodata_override=0)) == 0

    assert isnan(_resolve_dst_nodata(np.dtype("float32"), _cfg(), 0))
    assert _resolve_dst_nodata(np.dtype("uint16"), _cfg(), 0) == 0
    assert _resolve_dst_nodata(np.dtype("uint16"), _cfg(fill_value=3), 5) == 3
    assert _resolve_dst_nodata(np.dtype("float32"), _cfg(fill_value=3), 7) == 3


def test_resolve_dst_dtype():
    assert _resolve_dst_dtype("uint8", RasterLoadParams()) == "uint8"
    assert _resolve_dst_dtype("uint8", RasterLoadParams(dtype="float32")) == "float32"


def test_pick_overiew():
    assert _pick_overview(2, []) is None
    assert _pick_overview(1, [2, 4]) is None
    assert _pick_overview(2, [2, 4, 8]) == 0
    assert _pick_overview(3, [2, 4, 8]) == 0
    assert _pick_overview(4, [2, 4, 8]) == 1
    assert _pick_overview(7, [2, 4, 8]) == 1
    assert _pick_overview(8, [2, 4, 8]) == 2
    assert _pick_overview(20, [2, 4, 8]) == 2


def test_rio_read():
    gbox = GeoBox.from_bbox((-180, -90, 180, 90), shape=(160, 320), tight=True)

    non_zeros_roi = np.s_[30:47, 190:210]

    xx = xr_zeros(gbox, dtype="int16")
    xx.values[non_zeros_roi] = 333
    assert xx.odc.geobox == gbox

    cfg = RasterLoadParams()

    with with_temp_tiff(xx, compress=None) as uri:
        src = RasterSource(uri)

        # read whole
        roi, pix = rio_read(src, cfg, gbox)
        assert gbox[roi] == gbox
        assert pix.shape == gbox.shape
        assert_array_equal(pix, xx.values)

        # read part
        _gbox = gbox[non_zeros_roi]
        roi, pix = rio_read(src, cfg, _gbox)
        assert _gbox[roi] == _gbox
        assert pix.shape == _gbox.shape
        assert (pix == 333).all()
        assert_array_equal(pix, xx.values[non_zeros_roi])

        # - in-place dst
        # - dtype change to float32
        # - remap nodata to nan
        _cfg = RasterLoadParams(src_nodata_fallback=0)
        expect = xx.values.astype("float32")
        expect[xx.values == _cfg.src_nodata_fallback] = np.nan

        _dst = np.ones(gbox.shape, dtype="float32")
        roi, pix = rio_read(src, _cfg, gbox, dst=_dst)
        assert pix.dtype == _dst.dtype
        assert_array_equal(expect, pix)
        assert np.nansum(pix) == xx.values.sum()

        # - in-place dst
        # - remap nodata 0 -> -99
        _cfg = RasterLoadParams(src_nodata_fallback=0, fill_value=-99)
        expect = xx.values.copy()
        expect[xx.values == _cfg.src_nodata_fallback] = _cfg.fill_value

        _dst = np.ones(gbox.shape, dtype=xx.dtype)
        roi, pix = rio_read(src, _cfg, gbox, dst=_dst)
        assert pix.dtype == _dst.dtype
        assert (pix == _cfg.fill_value).any()
        assert (pix != _cfg.src_nodata_fallback).all()
        assert_array_equal(expect, pix)
        assert ma.masked_equal(pix, _cfg.fill_value).sum() == xx.values.sum()

    # smaller src than dst
    # float32 with nan
    _roi = np.s_[2:-2, 3:-5]
    _xx = xx.astype("float32").where(xx != 0, np.nan)[_roi]
    assert np.nansum(_xx.values) == xx.values.sum()
    assert _xx.odc.geobox == gbox[_roi]

    with with_temp_tiff(_xx, compress=None, overview_levels=[]) as uri:
        src = RasterSource(uri)

        # read whole input, filling only part of output
        roi, pix = rio_read(src, cfg, gbox)
        assert pix.shape == gbox[roi].shape
        assert gbox[roi] != gbox
        assert (gbox[roi] | gbox) == gbox
        assert_array_equal(pix, _xx.values)

    # smaller src than dst
    # no src nodata
    # yes dst nodata
    _xx = xx[_roi]
    assert _xx.values.sum() == xx.values.sum()
    assert _xx.odc.geobox == gbox[_roi]

    with with_temp_tiff(_xx, compress=None, overview_levels=[]) as uri:
        _cfg = RasterLoadParams(fill_value=-99)
        src = RasterSource(uri)

        # read whole input, filling only part of output
        roi, pix = rio_read(src, _cfg, gbox)
        assert pix.shape == gbox[roi].shape
        assert gbox[roi] != gbox
        assert (gbox[roi] | gbox) == gbox
        assert (pix != _cfg.fill_value).all()
        assert_array_equal(pix, _xx.values)

        # non-pasting path
        _gbox = gbox.zoom_out(1.3)
        roi, pix = rio_read(src, _cfg, _gbox)
        assert pix.shape == gbox[roi].shape
        assert gbox[roi] != gbox


def test_reader_ovr():
    ## smoke test only
    gbox = GeoBox.from_bbox((-180, -90, 180, 90), shape=(512, 512), tight=True)

    non_zeros_roi = np.s_[30:47, 190:210]

    xx = xr_zeros(gbox, dtype="int16")
    xx.values[non_zeros_roi] = 333
    assert xx.odc.geobox == gbox

    cfg = RasterLoadParams()

    # whole image from 1/2 overview
    with with_temp_tiff(xx, compress=None, overview_levels=[2, 4]) as uri:
        src = RasterSource(uri)
        _gbox = gbox.zoom_out(2)
        roi, pix = rio_read(src, cfg, _gbox)
        assert pix.shape == _gbox[roi].shape
        assert _gbox[roi] == _gbox


def test_reader_unhappy_paths():
    gbox = GeoBox.from_bbox((-180, -90, 180, 90), shape=(160, 320), tight=True)
    xx = xr_zeros(gbox, dtype="int16")

    with with_temp_tiff(xx, compress=None) as uri:
        cfg = RasterLoadParams()
        src = RasterSource(uri, band=3)

        # no such band error
        with pytest.raises(ValueError):
            _, _ = rio_read(src, cfg, gbox)


def test_reader_fail_on_error():
    gbox = GeoBox.from_bbox((-180, -90, 180, 90), shape=(160, 320), tight=True)
    xx = xr_zeros(gbox, dtype="int16")
    src = RasterSource("file:///no-such-path/no-such.tif")
    cfg = RasterLoadParams(dtype=str(xx.dtype), fail_on_error=True)

    # check that it raises error when fail_on_error=True
    with pytest.raises(rasterio.errors.RasterioIOError):
        _, _ = rio_read(src, cfg, gbox)

    # check that errors are suppressed when fail_on_error=False
    cfg = RasterLoadParams(dtype=str(xx.dtype), fail_on_error=False)
    roi, yy = rio_read(src, cfg, gbox)
    assert yy.shape == (0, 0)
    assert yy.dtype == cfg.dtype
    assert roi == np.s_[0:0, 0:0]

    roi, yy = rio_read(src, cfg, gbox, dst=xx.data)
    assert yy.shape == (0, 0)
    assert yy.dtype == cfg.dtype
    assert roi == np.s_[0:0, 0:0]
