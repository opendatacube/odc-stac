"""STAC Item -> ODC Dataset[eo3]."""
from ._version import __version__  # isort:skip  this has to be 1st import
from ._load import load
from ._mdtools import ConversionConfig
from ._model import (
    RasterBandMetadata,
    RasterCollectionMetadata,
    RasterLoadParams,
    RasterSource,
)

stac_load = load


__all__ = (
    "RasterBandMetadata",
    "RasterCollectionMetadata",
    "RasterLoadParams",
    "RasterSource",
    "ConversionConfig",
    "load",
    "stac_load",
    "__version__",
)
