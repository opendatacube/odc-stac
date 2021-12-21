"""STAC Item -> ODC Dataset[eo3]."""
from ._version import __version__  # isort:skip  this has to be 1st import
from ._dcload import configure_rio, dc_load
from ._eo3 import BandMetadata, ConversionConfig, infer_dc_product, stac2ds
from ._load import eo3_geoboxes, load

stac_load = load


__all__ = (
    "BandMetadata",
    "ConversionConfig",
    "stac2ds",
    "infer_dc_product",
    "configure_rio",
    "dc_load",
    "eo3_geoboxes",
    "load",
    "stac_load",
    "__version__",
)
