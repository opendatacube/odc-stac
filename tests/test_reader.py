from math import isnan

import numpy as np

from odc.stac._model import RasterLoadParams
from odc.stac._reader import (
    _pick_overview,
    _resolve_dst_dtype,
    _resolve_dst_nodata,
    _resolve_src_nodata,
    _same_nodata,
)


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
