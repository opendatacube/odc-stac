"""
Tools for constructing xarray objects from parsed metadata.
"""

from ._builder import chunked_load, resolve_chunk_shape
from ._driver import reader_driver, register_driver, unregister_driver
from ._reader import (
    resolve_dst_dtype,
    resolve_dst_nodata,
    resolve_load_cfg,
    resolve_src_nodata,
)
from .types import (
    BandIdentifier,
    BandKey,
    BandQuery,
    FixedCoord,
    MultiBandRasterSource,
    RasterBandMetadata,
    RasterGroupMetadata,
    RasterLoadParams,
    RasterSource,
    ReaderDriver,
)

__all__ = (
    "BandIdentifier",
    "BandKey",
    "BandQuery",
    "RasterBandMetadata",
    "RasterLoadParams",
    "RasterSource",
    "FixedCoord",
    "MultiBandRasterSource",
    "RasterGroupMetadata",
    "ReaderDriver",
    "chunked_load",
    "register_driver",
    "unregister_driver",
    "reader_driver",
    "resolve_load_cfg",
    "resolve_src_nodata",
    "resolve_dst_nodata",
    "resolve_dst_dtype",
    "resolve_chunk_shape",
)
