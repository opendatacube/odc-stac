"""
Utilities for reading pixels from raster files.

- nodata utilities
- read + reproject
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
from numpy.typing import DTypeLike

from .types import RasterBandMetadata, RasterLoadParams, with_default


def resolve_load_cfg(
    bands: dict[str, RasterBandMetadata],
    resampling: str | dict[str, str] | None = None,
    dtype: DTypeLike | dict[str, DTypeLike] | None = None,
    use_overviews: bool = True,
    nodata: float | None = None,
    fail_on_error: bool = True,
) -> dict[str, RasterLoadParams]:
    """
    Combine band metadata with user provided settings to produce load configuration.
    """

    def _dtype(name: str, band_dtype: str | None, fallback: str) -> str:
        if dtype is None:
            return with_default(band_dtype, fallback)
        if isinstance(dtype, dict):
            return str(
                with_default(
                    dtype.get(name, dtype.get("*", band_dtype)),
                    fallback,
                )
            )
        return str(dtype)

    def _resampling(name: str, fallback: str) -> str:
        if resampling is None:
            return fallback
        if isinstance(resampling, dict):
            return resampling.get(name, resampling.get("*", fallback))
        return resampling

    def _fill_value(band: RasterBandMetadata) -> float | None:
        if nodata is not None:
            return nodata
        return band.nodata

    def _resolve(name: str, band: RasterBandMetadata) -> RasterLoadParams:
        return RasterLoadParams(
            _dtype(name, band.data_type, "float32"),
            fill_value=_fill_value(band),
            use_overviews=use_overviews,
            resampling=_resampling(name, "nearest"),
            fail_on_error=fail_on_error,
        )

    return {name: _resolve(name, band) for name, band in bands.items()}


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


def pick_overview(read_shrink: int, overviews: Sequence[int]) -> Optional[int]:
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
