"""
STAC Item -> ODC Dataset[eo3]
"""
from ._version import __version__

from ._eo3 import (
    BandMetadata,
    ConversionConfig,
    stac2ds,
    infer_dc_product,
)

from ._dcload import dc_load
from ._load import load
stac_load = load


__all__ = (
    "BandMetadata",
    "ConversionConfig",
    "stac2ds",
    "infer_dc_product",
    "dc_load",
    "load",
    "stac_load",
    "__version__",
)
