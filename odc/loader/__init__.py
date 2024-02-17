"""
Tools for constructing xarray objects from parsed metadata.
"""

from ._builder import (
    DaskGraphBuilder,
    LoadChunkTask,
    MkArray,
    direct_chunked_load,
    fill_2d_slice,
    mk_dataset,
    resolve_chunk_shape,
)
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
    "LoadChunkTask",
    "DaskGraphBuilder",
    "mk_dataset",
    "MkArray",
    "fill_2d_slice",
    "direct_chunked_load",
    "reader_driver",
    "resolve_load_cfg",
    "resolve_chunk_shape",
)
