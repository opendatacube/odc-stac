"""STAC Item -> ODC Dataset[eo3]."""

from ._version import __version__  # isort:skip  this has to be 1st import
from odc.loader._rio import configure_rio, configure_s3_access
from odc.loader.types import RasterBandMetadata, RasterLoadParams, RasterSource

from ._mdtools import (
    ConversionConfig,
    ParsedItem,
    extract_collection_metadata,
    output_geobox,
    parse_item,
    parse_items,
)
from ._stac_load import load
from .model import RasterCollectionMetadata

stac_load = load


__all__ = (
    "ParsedItem",
    "RasterBandMetadata",
    "RasterCollectionMetadata",
    "RasterLoadParams",
    "RasterSource",
    "ConversionConfig",
    "load",
    "stac_load",
    "configure_rio",
    "configure_s3_access",
    "parse_item",
    "parse_items",
    "extract_collection_metadata",
    "output_geobox",
    "__version__",
)

_eo3_methods = ["stac2ds", "infer_dc_product"]


def __dir__():
    return [*__all__, *_eo3_methods]


def __getattr__(name):
    # pylint: disable=import-outside-toplevel
    if name in _eo3_methods:
        from . import eo3

        return getattr(eo3, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")
