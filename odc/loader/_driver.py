"""
Reader driver loader.

Currently always goes to rasterio
"""

from __future__ import annotations

from .types import RasterLoadParams, ReaderDriver


def reader_driver(cfg: dict[str, RasterLoadParams]) -> ReaderDriver:
    # pylint: disable=unused-argument, import-outside-toplevel
    from ._rio import RioDriver

    return RioDriver()
