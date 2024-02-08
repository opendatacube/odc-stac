"""
Utilities for reading pixels from raster files.

- nodata utilities
- read + reproject
"""

import math
from typing import List, Optional

import numpy as np

from ._model import RasterLoadParams


def resolve_src_nodata(
    nodata: Optional[float], cfg: RasterLoadParams
) -> Optional[float]:
    if cfg.src_nodata_override is not None:
        return cfg.src_nodata_override
    if nodata is not None:
        return nodata
    return cfg.src_nodata_fallback


def resolve_dst_dtype(src_dtype: str, cfg: RasterLoadParams) -> np.dtype:
    if cfg.dtype is None:
        return np.dtype(src_dtype)
    return np.dtype(cfg.dtype)


def resolve_dst_nodata(
    dst_dtype: np.dtype,
    cfg: RasterLoadParams,
    src_nodata: Optional[float] = None,
) -> Optional[float]:
    # 1. Configuration
    # 2. np.nan for float32 outputs
    # 3. Fall back to source nodata
    if cfg.fill_value is not None:
        return dst_dtype.type(cfg.fill_value)

    if dst_dtype.kind == "f":
        return np.nan

    if src_nodata is not None:
        return dst_dtype.type(src_nodata)

    return None


def pick_overview(read_shrink: int, overviews: List[int]) -> Optional[int]:
    if len(overviews) == 0 or read_shrink < overviews[0]:
        return None

    _idx = 0
    for idx, ovr in enumerate(overviews):
        if ovr > read_shrink:
            break
        _idx = idx

    return _idx


def same_nodata(a: Optional[float], b: Optional[float]) -> bool:
    if a is None:
        return b is None
    if b is None:
        return False
    if math.isnan(a):
        return math.isnan(b)
    return a == b


def nodata_mask(pix: np.ndarray, nodata: Optional[float]) -> np.ndarray:
    if pix.dtype.kind == "f":
        if nodata is None or math.isnan(nodata):
            return np.isnan(pix)
        return np.bitwise_or(np.isnan(pix), pix == nodata)
    if nodata is None:
        return np.zeros_like(pix, dtype="bool")
    return pix == nodata
