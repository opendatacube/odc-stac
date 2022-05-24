"""
Utilities for reading pixels from raster files.

- nodata utilities
- read + reproject
"""

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import rasterio
import rasterio.enums
import rasterio.warp
from odc.geo.geobox import GeoBox
from odc.geo.overlap import ReprojectInfo, compute_reproject_roi
from odc.geo.roi import NormalizedROI, roi_is_empty, roi_shape, w_
from odc.geo.warp import resampling_s2rio

from ._model import RasterLoadParams, RasterSource
from ._rio import rio_env


def _resolve_src_nodata(
    nodata: Optional[float], cfg: RasterLoadParams
) -> Optional[float]:
    if cfg.src_nodata_override is not None:
        return cfg.src_nodata_override
    if nodata is not None:
        return nodata
    return cfg.src_nodata_fallback


def _resolve_dst_dtype(src_dtype: str, cfg: RasterLoadParams) -> np.dtype:
    if cfg.dtype is None:
        return np.dtype(src_dtype)
    return np.dtype(cfg.dtype)


def _resolve_dst_nodata(
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


def _pick_overview(read_shrink: int, overviews: List[int]) -> Optional[int]:
    if len(overviews) == 0 or read_shrink < overviews[0]:
        return None

    _idx = 0
    for idx, ovr in enumerate(overviews):
        if ovr > read_shrink:
            break
        _idx = idx

    return _idx


def _rio_geobox(src: rasterio.DatasetReader) -> GeoBox:
    return GeoBox(src.shape, src.transform, src.crs)


def _same_nodata(a: Optional[float], b: Optional[float]) -> bool:
    if a is None:
        return b is None
    if b is None:
        return False
    if math.isnan(a):
        return math.isnan(b)
    return a == b


def _nodata_mask(pix: np.ndarray, nodata: Optional[float]) -> np.ndarray:
    if pix.dtype.kind == "f":
        if nodata is None or math.isnan(nodata):
            return np.isnan(pix)
        return np.bitwise_or(np.isnan(pix), pix == nodata)
    if nodata is None:
        return np.zeros_like(pix, dtype="bool")
    return pix == nodata


def _do_read(
    src: rasterio.Band,
    cfg: RasterLoadParams,
    dst_geobox: GeoBox,
    rr: ReprojectInfo,
    dst: Optional[np.ndarray] = None,
) -> Tuple[NormalizedROI, np.ndarray]:
    resampling = resampling_s2rio(cfg.resampling)
    rdr = src.ds

    if dst is not None:
        _dst = dst[rr.roi_dst]  # type: ignore
    else:
        _dst = np.ndarray(
            roi_shape(rr.roi_dst), dtype=_resolve_dst_dtype(src.dtype, cfg)
        )

    src_nodata0 = rdr.nodatavals[src.bidx - 1]
    src_nodata = _resolve_src_nodata(src_nodata0, cfg)
    dst_nodata = _resolve_dst_nodata(_dst.dtype, cfg, src_nodata)

    if roi_is_empty(rr.roi_dst):
        return (rr.roi_dst, _dst)

    if roi_is_empty(rr.roi_src):
        # no overlap case
        if dst_nodata is not None:
            np.copyto(_dst, dst_nodata)
        return (rr.roi_dst, _dst)

    if rr.paste_ok and rr.read_shrink == 1:
        rdr.read(src.bidx, out=_dst, window=w_[rr.roi_src])

        if dst_nodata is not None and not _same_nodata(src_nodata, dst_nodata):
            # remap nodata from source to output
            np.copyto(_dst, dst_nodata, where=_nodata_mask(_dst, src_nodata))
    else:
        # some form of reproject
        # TODO: support read with integer shrink then reproject more
        # TODO: deal with int8 inputs

        rasterio.warp.reproject(
            src,
            _dst,
            src_nodata=src_nodata,
            dst_crs=str(dst_geobox.crs),
            dst_transform=dst_geobox[rr.roi_dst].transform,
            dst_nodata=dst_nodata,
            resampling=resampling,
        )

    return (rr.roi_dst, _dst)


def src_geobox(src: Union[str, RasterSource]) -> GeoBox:
    """
    Get GeoBox of the source.

    1. If src is RasterSource with .geobox populated return that
    2. Else open file and read it
    """
    if isinstance(src, RasterSource):
        if src.geobox is not None:
            return src.geobox

        src = src.uri

    with rasterio.open(src, "r") as f:
        return _rio_geobox(f)


def rio_read(
    src: RasterSource,
    cfg: RasterLoadParams,
    dst_geobox: GeoBox,
    dst: Optional[np.ndarray] = None,
) -> Tuple[NormalizedROI, np.ndarray]:
    """
    Internal read method.

    Returns part of the destination image that overlaps with the given source.

    .. code-block: python

       cfg, geobox, sources = ...
       mosaic = np.full(geobox.shape, cfg.fill_value, dtype=cfg.dtype)

       for src in sources:
           roi, pix = rio_read(src, cfg, geobox)
           assert mosaic[roi].shape == pix.shape
           assert pix.dtype == mosaic.dtype

           # paste over destination pixels that are empty
           np.copyto(mosaic[roi], pix, where=(mosaic[roi] == nodata))
           # OR
           mosaic[roi] = pix  # if sources are true tiles (no overlaps)

    """
    # if resampling is `nearest` then ignore sub-pixel translation when deciding
    # whether we can just paste source into destination
    ttol = 0.9 if cfg.nearest else 0.05

    with rasterio.open(src.uri, "r", sharing=False) as rdr:
        assert isinstance(rdr, rasterio.DatasetReader)
        ovr_idx: Optional[int] = None

        if src.band > rdr.count:
            raise ValueError(f"No band {src.band} in '{src.uri}'")

        rr = compute_reproject_roi(_rio_geobox(rdr), dst_geobox, ttol=ttol)

        if cfg.use_overviews and rr.read_shrink > 1:
            ovr_idx = _pick_overview(rr.read_shrink, rdr.overviews(src.band))

        if ovr_idx is None:
            with rio_env(VSI_CACHE=False):
                return _do_read(
                    rasterio.band(rdr, src.band), cfg, dst_geobox, rr, dst=dst
                )

        # read from overview
        with rasterio.open(
            src.uri, "r", sharing=False, overview_level=ovr_idx
        ) as rdr_ovr:
            rr = compute_reproject_roi(_rio_geobox(rdr_ovr), dst_geobox, ttol=ttol)
            with rio_env(VSI_CACHE=False):
                return _do_read(
                    rasterio.band(rdr_ovr, src.band), cfg, dst_geobox, rr, dst=dst
                )
