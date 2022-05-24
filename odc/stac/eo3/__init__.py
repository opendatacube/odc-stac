"""
Datacube metadata generation from STAC.

This only imports if datacube is installed.
"""
from ._eo3converter import infer_dc_product, stac2ds

__all__ = (
    "stac2ds",
    "infer_dc_product",
)
