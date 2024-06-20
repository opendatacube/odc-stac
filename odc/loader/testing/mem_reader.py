"""
Reader Driver from in-memory xarray.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.xr import ODCExtensionDa, ODCExtensionDs, xr_reproject

from ..types import (
    BandKey,
    DaskRasterReader,
    FixedCoord,
    MDParser,
    RasterBandMetadata,
    RasterGroupMetadata,
    RasterLoadParams,
    RasterSource,
    ReaderSubsetSelection,
)


class XrMDPlugin:
    """
    Convert xarray.Dataset to RasterGroupMetadata.

    Implements MDParser interface.

    - Convert xarray.Dataset to RasterGroupMetadata
    - Driver data is xarray.DataArray for each band
    """

    def __init__(self, src: xr.Dataset) -> None:
        self._src = src
        self._md = raster_group_md(src)

    def extract(self, md: Any) -> RasterGroupMetadata:
        """Fixed description of src dataset."""
        assert md is not None
        return self._md

    def driver_data(self, md: Any, band_key: BandKey) -> xr.DataArray:
        """
        Extract driver specific data for a given band.
        """
        assert md is not None
        name, _ = band_key
        return self._src[name]


class Context:
    """Context shared across a single load operation."""

    # pylint: disable=too-few-public-methods

    def __init__(
        self,
        src: xr.Dataset,
        geobox: GeoBox,
        chunks: None | dict[str, int],
    ) -> None:
        self.src = src
        self.geobox = geobox
        self.chunks = chunks

    def with_env(self, env: dict[str, Any]) -> "Context":
        assert isinstance(env, dict)
        return Context(self.src, self.geobox, self.chunks)


class XrMemReader:
    """
    Protocol for raster readers.
    """

    # pylint: disable=too-few-public-methods

    def __init__(self, src: RasterSource, ctx: Context) -> None:
        self._src = src
        self._xx: xr.DataArray = src.driver_data
        self._ctx = ctx

    def read(
        self,
        cfg: RasterLoadParams,
        dst_geobox: GeoBox,
        *,
        dst: np.ndarray | None = None,
        selection: ReaderSubsetSelection | None = None,
    ) -> tuple[tuple[slice, slice], np.ndarray]:
        src = self._xx

        if selection is not None:
            # only support single extra dimension
            assert isinstance(selection, (slice, int)) or len(selection) == 1
            assert len(cfg.extra_dims) == 1
            (band_dim,) = cfg.extra_dims
            src = src.isel({band_dim: selection})

        warped = xr_reproject(src, dst_geobox, resampling=cfg.resampling)
        assert isinstance(warped.data, np.ndarray)

        if dst is None:
            dst = warped.data
        else:
            dst[...] = warped.data

        yx_roi = (slice(None), slice(None))
        return yx_roi, dst


class XrMemReaderDriver:
    """
    Read from in memory xarray.Dataset.
    """

    Reader = XrMemReader

    def __init__(self, src: xr.Dataset) -> None:
        self.src = src

    def new_load(
        self,
        geobox: GeoBox,
        *,
        chunks: None | dict[str, int] = None,
    ) -> Context:
        return Context(self.src, geobox, chunks)

    def finalise_load(self, load_state: Context) -> Context:
        return load_state

    def capture_env(self) -> dict[str, Any]:
        return {}

    @contextmanager
    def restore_env(
        self, env: dict[str, Any], load_state: Context
    ) -> Iterator[Context]:
        yield load_state.with_env(env)

    def open(self, src: RasterSource, ctx: Context) -> XrMemReader:
        return XrMemReader(src, ctx)

    @property
    def md_parser(self) -> MDParser:
        return XrMDPlugin(self.src)

    @property
    def dask_reader(self) -> DaskRasterReader | None:
        return None


def band_info(xx: xr.DataArray) -> RasterBandMetadata:
    """
    Extract band metadata from xarray.DataArray
    """
    oo: ODCExtensionDa = xx.odc
    ydim = oo.ydim

    if xx.ndim > 2:
        dims = tuple(str(d) for d in xx.dims)
        dims = dims[:ydim] + ("y", "x") + dims[ydim + 2 :]
    else:
        dims = ()

    return RasterBandMetadata(
        data_type=str(xx.dtype),
        nodata=oo.nodata,
        units=xx.attrs.get("units", "1"),
        dims=dims,
    )


def raster_group_md(src: xr.Dataset) -> RasterGroupMetadata:
    oo: ODCExtensionDs = src.odc
    sdims = oo.spatial_dims or ("y", "x")

    bands: dict[BandKey, RasterBandMetadata] = {
        (str(k), 1): band_info(v) for k, v in src.data_vars.items() if v.ndim >= 2
    }

    extra_dims: dict[str, int] = {
        str(name): sz for name, sz in src.sizes.items() if name not in sdims
    }

    aliases: dict[str, list[BandKey]] = {}

    extra_coords: list[FixedCoord] = []
    for coord in src.coords.values():
        if len(coord.dims) != 1 or coord.dims[0] in sdims:
            # Only 1-d non-spatial coords
            continue
        extra_coords.append(
            FixedCoord(
                coord.name,
                coord.values,
                dim=coord.dims[0],
                units=coord.attrs.get("units", "1"),
            )
        )

    return RasterGroupMetadata(bands, aliases, extra_dims, extra_coords)
