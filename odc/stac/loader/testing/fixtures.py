"""
Test fixture construction utilities.
"""

from __future__ import annotations

import atexit
import os
import pathlib
import shutil
import tempfile
from collections import abc
from contextlib import contextmanager
from typing import Any, Generator, Sequence, Tuple

import rasterio
import xarray as xr
from odc.geo.xr import ODCExtensionDa

from ..types import RasterBandMetadata


@contextmanager
def with_temp_tiff(data: xr.DataArray, **cog_opts) -> Generator[str, None, None]:
    """
    Dump array to temp image and return path to it.
    """
    assert isinstance(data.odc, ODCExtensionDa)

    with rasterio.MemoryFile() as mem:
        data.odc.write_cog(mem.name, **cog_opts)  # type: ignore
        yield mem.name


def write_files(file_dict):
    """
    Convenience method for writing a bunch of files to a temporary directory.

    Dict format is "filename": "text content"

    If content is another dict, it is created recursively in the same manner.

    writeFiles({'test.txt': 'contents of text file'})

    :return: Created temporary directory path
    """
    containing_dir = tempfile.mkdtemp(suffix="testrun")
    _write_files_to_dir(containing_dir, file_dict)

    def remove_if_exists(path):
        if os.path.exists(path):
            shutil.rmtree(path)

    atexit.register(remove_if_exists, containing_dir)
    return pathlib.Path(containing_dir)


def _write_files_to_dir(directory_path, file_dict):
    """
    Convenience method for writing a bunch of files to a given directory.
    """
    for filename, contents in file_dict.items():
        path = os.path.join(directory_path, filename)
        if isinstance(contents, abc.Mapping):
            os.mkdir(path)
            _write_files_to_dir(path, contents)
        else:
            with open(path, "w", encoding="utf8") as f:
                if isinstance(contents, str):
                    f.write(contents)
                elif isinstance(contents, abc.Sequence):
                    f.writelines(contents)
                else:
                    raise ValueError(f"Unexpected file contents: {type(contents)}")


class FakeMDPlugin:
    """
    Fake metadata extraction plugin for testing.
    """

    def __init__(
        self,
        bands: Sequence[RasterBandMetadata] | dict[str, Sequence[RasterBandMetadata]],
        aliases: Sequence[str] | dict[str, Sequence[str]],
        driver_data,
    ):
        self._bands = bands
        self._aliases = aliases
        self._driver_data = driver_data

    def aliases(self, md, name: str) -> Tuple[str, ...]:
        assert md is not None
        if isinstance(self._aliases, dict):
            return tuple(self._aliases.get(name, ()))
        return tuple(self._aliases)

    def bands(self, md, name: str) -> Tuple[RasterBandMetadata, ...]:
        assert md is not None
        if isinstance(self._bands, dict):
            return tuple(self._bands.get(name, ()))
        return tuple(self._bands)

    def driver_data(self, md, name: str, idx: int) -> Any:
        assert md is not None
        if isinstance(self._driver_data, dict):
            if name in self._driver_data:
                return self._driver_data[name]
            if (name, idx) in self._driver_data:
                return self._driver_data[name, idx]
        return self._driver_data
