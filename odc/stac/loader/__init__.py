"""
Tools for constructing xarray objects from parsed metadata.
"""

from ._builder import (
    DaskGraphBuilder,
    LoadChunkTask,
    MkArray,
    fill_2d_slice,
    mk_dataset,
)
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
)
