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
from .types import RasterBandMetadata, RasterLoadParams, RasterSource, SomeReader

__all__ = (
    "RasterBandMetadata",
    "RasterLoadParams",
    "RasterSource",
    "SomeReader",
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
