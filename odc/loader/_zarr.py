"""
Reader Driver from in-memory xarray/zarr spec docs.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Iterator, Mapping, Sequence

import fsspec
import numpy as np
import xarray as xr
from dask.delayed import Delayed, delayed
from odc.geo.geobox import GeoBox
from odc.geo.xr import ODCExtensionDa, ODCExtensionDs, xr_coords, xr_reproject

from .types import (
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

# TODO: tighten specs for Zarr*
SomeDoc = Mapping[str, Any]
ZarrSpec = Mapping[str, Any]
ZarrSpecFs = Mapping[str, Any]
ZarrSpecFsDict = dict[str, Any]
# pylint: disable=too-few-public-methods


def extract_zarr_spec(src: SomeDoc) -> ZarrSpecFsDict | None:
    if ".zmetadata" in src:
        return dict(src)

    if "zarr:metadata" in src:
        # TODO: handle zarr:chunks for reference filesystem
        zmd = {"zarr_consolidated_format": 1, "metadata": src["zarr:metadata"]}
    elif "zarr_consolidated_format" in src:
        zmd = dict(src)
    else:
        zmd = {"zarr_consolidated_format": 1, "metadata": src}

    return {".zmetadata": json.dumps(zmd)}


def _from_zarr_spec(
    spec_doc: ZarrSpecFs,
    regen_coords: bool = False,
    fs: fsspec.AbstractFileSystem | None = None,
    chunks=None,
    target: str | None = None,
    fsspec_opts: dict[str, Any] | None = None,
) -> xr.Dataset:
    fsspec_opts = fsspec_opts or {}
    rfs = fsspec.filesystem(
        "reference", fo=spec_doc, fs=fs, target=target, **fsspec_opts
    )

    xx = xr.open_dataset(rfs.get_mapper(""), engine="zarr", mode="r", chunks=chunks)
    gbox = xx.odc.geobox
    if gbox is not None and regen_coords:
        # re-gen x,y coords from geobox
        xx = xx.assign_coords(xr_coords(gbox))

    return xx


def _resolve_src_dataset(
    md: Any,
    *,
    regen_coords: bool = False,
    fallback: xr.Dataset | None = None,
    **kw,
) -> xr.Dataset | None:
    if isinstance(md, dict) and (spec_doc := extract_zarr_spec(md)) is not None:
        return _from_zarr_spec(spec_doc, regen_coords=regen_coords, **kw)

    if isinstance(md, xr.Dataset):
        return md

    # TODO: support stac items and datacube datasets
    return fallback


class XrMDPlugin:
    """
    Convert xarray.Dataset to RasterGroupMetadata.

    Implements MDParser interface.

    - Convert xarray.Dataset to RasterGroupMetadata
    - Driver data is xarray.DataArray for each band
    """

    def __init__(
        self,
        template: RasterGroupMetadata,
        fallback: xr.Dataset | None = None,
    ) -> None:
        self._template = template
        self._fallback = fallback

    def _resolve_src(self, md: Any, regen_coords: bool = False) -> xr.Dataset | None:
        return _resolve_src_dataset(
            md, regen_coords=regen_coords, fallback=self._fallback, chunks={}
        )

    def extract(self, md: Any) -> RasterGroupMetadata:
        """Fixed description of src dataset."""
        if isinstance(md, RasterGroupMetadata):
            return md

        if (src := self._resolve_src(md, regen_coords=False)) is not None:
            return raster_group_md(src, base=self._template)

        return self._template

    def driver_data(self, md: Any, band_key: BandKey) -> xr.DataArray | SomeDoc | None:
        """
        Extract driver specific data for a given band.
        """
        name, _ = band_key

        if isinstance(md, dict):
            if (spec_doc := extract_zarr_spec(md)) is not None:
                return spec_doc
            return md

        if isinstance(md, xr.DataArray):
            return md

        src = self._resolve_src(md, regen_coords=False)
        if src is None or name not in src.data_vars:
            return None

        return src.data_vars[name]


class Context:
    """Context shared across a single load operation."""

    # pylint: disable=too-few-public-methods

    def __init__(
        self,
        geobox: GeoBox,
        chunks: None | dict[str, int],
        driver: Any | None = None,
    ) -> None:
        self.geobox = geobox
        self.chunks = chunks
        self.driver = driver

    def with_env(self, env: dict[str, Any]) -> "Context":
        assert isinstance(env, dict)
        return Context(self.geobox, self.chunks)


class XrSource:
    """
    RasterSource -> xr.DataArray|xr.Dataset
    """

    def __init__(self, src: RasterSource, chunks: Any | None = None) -> None:
        driver_data: xr.DataArray | xr.Dataset | SomeDoc = src.driver_data
        self._spec: ZarrSpecFs | None = None
        self._ds: xr.Dataset | None = None
        self._xx: xr.DataArray | None = None
        self._src = src
        self._chunks = chunks

        if isinstance(driver_data, xr.DataArray):
            self._xx = driver_data
        elif isinstance(driver_data, xr.Dataset):
            subdataset = src.subdataset
            self._ds = driver_data
            assert subdataset in driver_data.data_vars
            self._xx = driver_data.data_vars[subdataset]
        elif isinstance(driver_data, dict):
            self._spec = extract_zarr_spec(driver_data)
        elif driver_data is not None:
            raise ValueError(f"Unsupported driver data type: {type(driver_data)}")

        assert driver_data is None or (self._spec is not None or self._xx is not None)

    @property
    def spec(self) -> ZarrSpecFs | None:
        return self._spec

    def base(self, regen_coords: bool = False) -> xr.Dataset | None:
        if self._ds is not None:
            return self._ds
        if self._spec is None:
            return None
        self._ds = _from_zarr_spec(
            self._spec,
            regen_coords=regen_coords,
            target=self._src.uri,
            chunks=self._chunks,
        )
        return self._ds

    def resolve(
        self,
        regen_coords: bool = False,
    ) -> xr.DataArray:
        if self._xx is not None:
            return self._xx

        src_ds = self.base(regen_coords=regen_coords)
        if src_ds is None:
            raise ValueError("Failed to interpret driver data")

        subdataset = self._src.subdataset
        if subdataset is None:
            _first, *_ = src_ds.data_vars
            subdataset = str(_first)

        if subdataset not in src_ds.data_vars:
            raise ValueError(f"Band {subdataset!r} not found in dataset")

        self._xx = src_ds.data_vars[subdataset]
        return self._xx


def _subset_src(
    src: xr.DataArray, selection: ReaderSubsetSelection, cfg: RasterLoadParams
) -> xr.DataArray:
    if selection is None:
        return src

    assert isinstance(selection, (slice, int)) or len(selection) == 1
    assert len(cfg.extra_dims) == 1
    (band_dim,) = cfg.extra_dims
    return src.isel({band_dim: selection})


class XrMemReader:
    """
    Implements protocol for raster readers.

    - Read from in-memory xarray.Dataset
    - Read from zarr spec
    """

    def __init__(self, src: RasterSource, ctx: Context) -> None:
        self._src = XrSource(src, chunks=None)
        self._ctx = ctx

    def read(
        self,
        cfg: RasterLoadParams,
        dst_geobox: GeoBox,
        *,
        dst: np.ndarray | None = None,
        selection: ReaderSubsetSelection | None = None,
    ) -> tuple[tuple[slice, slice], np.ndarray]:
        src = self._src.resolve(regen_coords=True)
        src = _subset_src(src, selection, cfg)

        warped = xr_reproject(src, dst_geobox, resampling=cfg.resampling)
        assert isinstance(warped.data, np.ndarray)

        if dst is None:
            dst = warped.data
        else:
            dst[...] = warped.data

        yx_roi = (slice(None), slice(None))
        return yx_roi, dst


def _with_roi(xx: np.ndarray) -> tuple[tuple[slice, slice], np.ndarray]:
    return (slice(None), slice(None)), xx


class XrMemReaderDask:
    """
    Dask version of the reader.
    """

    def __init__(
        self,
        src: RasterSource | None = None,
        ctx: Context | None = None,
        layer_name: str = "",
        idx: int = -1,
    ) -> None:
        self._src = XrSource(src, chunks={}) if src is not None else None
        self._ctx = ctx
        self._layer_name = layer_name
        self._idx = idx

    def read(
        self,
        cfg: RasterLoadParams,
        dst_geobox: GeoBox,
        *,
        selection: ReaderSubsetSelection | None = None,
        idx: tuple[int, ...] = (),
    ) -> Delayed:
        assert self._src is not None
        assert isinstance(idx, tuple)

        xx = self._src.resolve(regen_coords=True)
        xx = _subset_src(xx, selection, cfg)
        yy = xr_reproject(
            xx,
            dst_geobox,
            resampling=cfg.resampling,
            chunks=dst_geobox.shape.yx,
        )
        return delayed(_with_roi)(yy.data, dask_key_name=(self._layer_name, *idx))

    def open(
        self,
        src: RasterSource,
        ctx: Any,
        *,
        layer_name: str,
        idx: int,
    ) -> DaskRasterReader:
        return XrMemReaderDask(src, ctx, layer_name=layer_name, idx=idx)


class XrMemReaderDriver:
    """
    Read from in memory xarray.Dataset or zarr spec document.
    """

    Reader = XrMemReader

    def __init__(
        self,
        src: xr.Dataset | None = None,
        template: RasterGroupMetadata | None = None,
    ) -> None:
        if src is not None and template is None:
            template = raster_group_md(src)
        if template is None:
            template = RasterGroupMetadata({}, {}, {}, [])
        self.src = src
        self.template = template

    def new_load(
        self,
        geobox: GeoBox,
        *,
        chunks: None | dict[str, int] = None,
    ) -> Context:
        return Context(geobox, chunks, driver=self)

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
        return XrMDPlugin(self.template, fallback=self.src)

    @property
    def dask_reader(self) -> DaskRasterReader | None:
        return XrMemReaderDask()


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


def raster_group_md(
    src: xr.Dataset,
    *,
    base: RasterGroupMetadata | None = None,
    aliases: dict[str, list[BandKey]] | None = None,
    extra_coords: Sequence[FixedCoord] = (),
    extra_dims: dict[str, int] | None = None,
) -> RasterGroupMetadata:
    oo: ODCExtensionDs = src.odc
    sdims = oo.spatial_dims or ("y", "x")

    if base is None:
        base = RasterGroupMetadata(
            bands={},
            aliases=aliases or {},
            extra_coords=extra_coords,
            extra_dims=extra_dims or {},
        )

    bands = base.bands.copy()
    bands.update(
        {(str(k), 1): band_info(v) for k, v in src.data_vars.items() if v.ndim >= 2}
    )

    edims = base.extra_dims.copy()
    edims.update({str(name): sz for name, sz in src.sizes.items() if name not in sdims})

    aliases: dict[str, list[BandKey]] = base.aliases.copy()

    extra_coords: list[FixedCoord] = list(base.extra_coords)
    supplied_coords = set(coord.name for coord in extra_coords)

    for coord in src.coords.values():
        if len(coord.dims) != 1 or coord.dims[0] in sdims:
            # Only 1-d non-spatial coords
            continue

        if coord.name in supplied_coords:
            continue

        extra_coords.append(
            FixedCoord(
                coord.name,
                coord.values,
                dim=coord.dims[0],
                units=coord.attrs.get("units", "1"),
            )
        )

    return RasterGroupMetadata(
        bands=bands,
        aliases=aliases,
        extra_dims=edims,
        extra_coords=extra_coords,
    )
