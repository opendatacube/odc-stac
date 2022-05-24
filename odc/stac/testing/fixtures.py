"""
Test fixture construction utilities.
"""
import atexit
import os
import pathlib
import shutil
import tempfile
from collections.abc import Mapping, Sequence
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
        if isinstance(contents, Mapping):
            os.mkdir(path)
            _write_files_to_dir(path, contents)
        else:
            with open(path, "w", encoding="utf8") as f:
                if isinstance(contents, str):
                    f.write(contents)
                elif isinstance(contents, Sequence):
                    f.writelines(contents)
                else:
                    raise ValueError(f"Unexpected file contents: {type(contents)}")
