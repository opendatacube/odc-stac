"""
Reader driver loader.

Currently always goes to rasterio
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Callable

from ._rio import RioDriver
from ._zarr import XrMemReaderDriver
from .types import ReaderDriver, ReaderDriverSpec, is_reader_driver

_available_drivers: dict[str, Callable[[], ReaderDriver] | ReaderDriver] = {
    "rio": RioDriver,
    "zarr": XrMemReaderDriver,
}


def register_driver(
    name: str, driver: Callable[[], ReaderDriver] | ReaderDriver, /
) -> None:
    """
    Register a new driver
    """
    _available_drivers[name] = driver


def unregister_driver(name: str, /) -> None:
    """
    Unregister a driver
    """
    _available_drivers.pop(name, None)


def _norm_driver(drv: Any) -> ReaderDriver:
    if isinstance(drv, type):
        return drv()
    if is_reader_driver(drv):
        return drv
    return drv()


def reader_driver(x: ReaderDriverSpec | None = None, /) -> ReaderDriver:
    if x is None:
        return RioDriver()
    if not isinstance(x, str):
        return x

    if (drv := _available_drivers.get(x)) is not None:
        return _norm_driver(drv)

    if "." not in x:
        raise ValueError(f"Unknown driver: {x!r}")

    module_name, class_name = x.rsplit(".", 1)
    try:
        cls = getattr(import_module(module_name), class_name)
        return cls()
    except (ModuleNotFoundError, AttributeError):
        raise ValueError(f"Failed to resolve driver spec: {x!r}") from None
