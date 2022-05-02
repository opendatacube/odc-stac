"""
Test fixture construction utilities.
"""
from contextlib import contextmanager
from typing import Generator

import rasterio
import xarray as xr
from odc.geo.xr import ODCExtensionDa


@contextmanager
def with_temp_tiff(data: xr.DataArray, **cog_opts) -> Generator[str, None, None]:
    """
    Dump array to temp image and return path to it.
    """
    assert isinstance(data.odc, ODCExtensionDa)

    with rasterio.MemoryFile() as mem:
        data.odc.write_cog(mem.name, **cog_opts)
        yield mem.name
