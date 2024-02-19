"""
Tools for constructing xarray objects from parsed metadata.
"""

from ._builder import chunked_load, resolve_chunk_shape
from ._driver import reader_driver
from ._reader import resolve_load_cfg
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
    "reader_driver",
    "resolve_load_cfg",
    "resolve_chunk_shape",
)
