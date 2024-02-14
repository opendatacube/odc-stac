"""
Reader driver loader.

Currently always goes to rasterio
"""

from __future__ import annotations

from .types import RasterLoadParams, SomeReader


def reader_driver(cfg: dict[str, RasterLoadParams]) -> SomeReader:
    # pylint: disable=unused-argument, import-outside-toplevel
    from ._rio import RioReader

    return RioReader()
