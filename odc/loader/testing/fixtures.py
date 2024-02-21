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
from typing import Any, Dict, Generator, Iterator, Optional, Tuple

import numpy as np
import rasterio
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.roi import NormalizedROI
from odc.geo.xr import ODCExtensionDa

from ..types import (
    BandKey,
    MDParser,
    RasterGroupMetadata,
    RasterLoadParams,
    RasterSource,
)

# pylint: disable=too-few-public-methods


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
        group_md: RasterGroupMetadata,
        driver_data,
    ):
        self._group_md = group_md
        self._driver_data = driver_data

    def extract(self, md: Any) -> RasterGroupMetadata:
        assert md is not None
        return self._group_md

    def driver_data(self, md, band_key: BandKey) -> Any:
        assert md is not None
        name, _ = band_key
        if isinstance(self._driver_data, dict):
            if name in self._driver_data:
                return self._driver_data[name]
            if band_key in self._driver_data:
                return self._driver_data[band_key]
        return self._driver_data


class FakeReader:
    """
    Fake reader for testing.
    """

    class LoadState:
        """
        Shared state for all readers for a given load.
        """

        def __init__(
            self, group_md: RasterGroupMetadata, env: dict[str, Any], is_dask: bool
        ) -> None:
            self.group_md = group_md
            self.env = env
            self.is_dask = is_dask
            self.finalised = False

        def with_env(self, env: dict[str, Any]) -> "FakeReader.LoadState":
            return FakeReader.LoadState(self.group_md, env, self.is_dask)

    def __init__(self, src: RasterSource, load_state: "FakeReader.LoadState"):
        self._src = src
        self._load_state = load_state

    def _extra_dims(self) -> Dict[str, int]:
        md = self._load_state.group_md
        return md.extra_dims or {
            coord.dim: len(coord.values) for coord in md.extra_coords
        }

    def read(
        self,
        cfg: RasterLoadParams,
        dst_geobox: GeoBox,
        dst: Optional[np.ndarray] = None,
    ) -> Tuple[NormalizedROI, np.ndarray]:
        meta = self._src.meta
        assert meta is not None

        extra_dims = self._extra_dims()
        postfix_dims: Tuple[int, ...] = ()
        if meta.dims is not None:
            assert set(meta.dims[2:]).issubset(extra_dims)
            postfix_dims = tuple(extra_dims[d] for d in meta.dims[2:])

        ny, nx = dst_geobox.shape.yx
        yx_roi = (slice(0, ny), slice(0, nx))
        shape = (ny, nx, *postfix_dims)

        src_pix: np.ndarray | None = self._src.driver_data
        if src_pix is None:
            src_pix = np.ones(shape, dtype=cfg.dtype)
        else:
            assert src_pix.shape == shape

        assert postfix_dims == src_pix.shape[2:]

        if dst is None:
            dst = np.zeros((ny, nx, *postfix_dims), dtype=cfg.dtype)
            dst[:] = src_pix.astype(dst.dtype)
            return yx_roi, dst

        assert dst.shape == src_pix.shape
        dst[:] = src_pix.astype(dst.dtype)

        return yx_roi, dst[yx_roi]


class FakeReaderDriver:
    """
    Fake reader for testing.
    """

    def __init__(
        self,
        group_md: RasterGroupMetadata,
        *,
        parser: MDParser | None = None,
    ):
        self._group_md = group_md
        self._parser = parser or FakeMDPlugin(group_md, None)

    def new_load(self, chunks: None | Dict[str, int] = None) -> FakeReader.LoadState:
        return FakeReader.LoadState(self._group_md, {}, chunks is not None)

    def finalise_load(self, load_state: FakeReader.LoadState) -> Any:
        assert load_state.finalised is False
        load_state.finalised = True
        return load_state

    def capture_env(self) -> Dict[str, Any]:
        return {}

    @contextmanager
    def restore_env(
        self, env: Dict[str, Any], load_state: FakeReader.LoadState
    ) -> Iterator[FakeReader.LoadState]:
        yield load_state.with_env(env)

    def open(self, src: RasterSource, ctx: FakeReader.LoadState) -> FakeReader:
        return FakeReader(src, ctx)

    @property
    def md_parser(self) -> MDParser | None:
        return self._parser
