"""
Utilities for reading pixels from raster files.

- nodata utilities
- read + reproject
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import numpy as np
from dask import delayed
from odc.geo.geobox import GeoBox

from .types import (
    Band_DType,
    RasterBandMetadata,
    RasterLoadParams,
    RasterSource,
    ReaderDriver,
    ReaderSubsetSelection,
    with_default,
)


def _dask_read_adaptor(
    src: RasterSource,
    ctx: Any,
    cfg: RasterLoadParams,
    dst_geobox: GeoBox,
    driver: ReaderDriver,
    env: dict[str, Any],
    selection: Optional[ReaderSubsetSelection] = None,
) -> tuple[tuple[slice, slice], np.ndarray]:

    with driver.restore_env(env, ctx) as local_ctx:
        rdr = driver.open(src, local_ctx)
        return rdr.read(cfg, dst_geobox, selection=selection)


class ReaderDaskAdaptor:
    """
    Creates default ``DaskRasterReader`` from a ``ReaderDriver``.

    Suitable for implementing ``.dask_reader`` property for generic reader drivers.
    """

    def __init__(
        self,
        driver: ReaderDriver,
        env: dict[str, Any] | None = None,
        ctx: Any | None = None,
        src: RasterSource | None = None,
        cfg: RasterLoadParams | None = None,
        layer_name: str = "",
        idx: int = -1,
    ) -> None:
        if env is None:
            env = driver.capture_env()

        self._driver = driver
        self._env = env
        self._ctx = ctx
        self._src = src
        self._cfg = cfg
        self._layer_name = layer_name
        self._src_idx = idx

    def read(
        self,
        dst_geobox: GeoBox,
        *,
        selection: Optional[ReaderSubsetSelection] = None,
        idx: tuple[int, ...],
    ) -> Any:
        assert self._src is not None
        assert self._ctx is not None
        assert self._cfg is not None

        read_op = delayed(_dask_read_adaptor, name=self._layer_name)

        # TODO: supply `dask_key_name=` that makes sense
        return read_op(
            self._src,
            self._ctx,
            self._cfg,
            dst_geobox,
            self._driver,
            self._env,
            selection=selection,
            dask_key_name=(self._layer_name, *idx),
        )

    def open(
        self,
        src: RasterSource,
        cfg: RasterLoadParams,
        ctx: Any,
        layer_name: str,
        idx: int,
    ) -> "ReaderDaskAdaptor":
        return ReaderDaskAdaptor(
            self._driver,
            self._env,
            ctx,
            src,
            cfg,
            layer_name=layer_name,
            idx=idx,
        )


def resolve_load_cfg(
    bands: dict[str, RasterBandMetadata],
    resampling: str | dict[str, str] | None = None,
    dtype: Band_DType | None = None,
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
            dims=band.dims,
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


def resolve_dst_fill_value(
    dst_dtype: np.dtype,
    cfg: RasterLoadParams,
    src_nodata: Optional[float] = None,
) -> float:
    nodata = resolve_dst_nodata(dst_dtype, cfg, src_nodata)
    if nodata is None:
        return dst_dtype.type(0)
    return nodata


def _selection_to_bands(selection: Any, n: int) -> list[int]:
    if selection is None:
        return list(range(1, n + 1))

    if isinstance(selection, list):
        return selection

    bidx = np.arange(1, n + 1)
    if isinstance(selection, int):
        return [int(bidx[selection])]
    return bidx[selection].tolist()


def resolve_band_query(
    src: RasterSource,
    n: int,
    selection: ReaderSubsetSelection | None = None,
) -> int | list[int]:
    if src.band > n:
        raise ValueError(
            f"Requested band {src.band} from {src.uri} with only {n} bands"
        )

    if src.band == 0:
        return _selection_to_bands(selection, n)

    meta = src.meta
    if meta is None:
        return src.band
    if meta.extra_dims:
        return [src.band]

    return src.band


def expand_selection(selection: Any, ydim: int) -> tuple[slice, ...]:
    """
    Add Y/X slices to selection tuple

    :param selection: Selection object
    :return: Tuple of slices
    """
    if selection is None:
        selection = ()
    if not isinstance(selection, tuple):
        selection = (selection,)

    prefix, postfix = selection[:ydim], selection[ydim:]
    return prefix + (slice(None), slice(None)) + postfix


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
