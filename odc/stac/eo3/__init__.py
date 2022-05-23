"""
Datacube metadata generation from STAC.

This only imports if datacube is installed.
"""
from ._dcload import configure_rio, dc_load
from ._eo3converter import infer_dc_product, stac2ds
from ._load_via_dc import eo3_geoboxes, load

__all__ = (
    "stac2ds",
    "infer_dc_product",
    "configure_rio",
    "dc_load",
    "load",
    "eo3_geoboxes",
)
