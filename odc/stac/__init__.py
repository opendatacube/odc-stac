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


__all__ = (
    "BandMetadata",
    "ConversionConfig",
    "stac2ds",
    "infer_dc_product",
    "dc_load",
    "__version__",
)
