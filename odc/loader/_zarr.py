"""
Reader Driver from in-memory xarray/zarr spec docs.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from typing import Any, Iterator

import dask.array as da
import fsspec
import numpy as np
import xarray as xr
from dask import is_dask_collection
from dask.array.core import normalize_chunks
from dask.delayed import Delayed, delayed
from fsspec.core import url_to_fs
from odc.geo.geobox import GeoBox, GeoBoxBase
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
ZarrSpecDict = dict[str, Any]
# pylint: disable=too-few-public-methods


def extract_zarr_spec(src: SomeDoc) -> ZarrSpecDict | None:
    if ".zgroup" in src:
        return dict(src)

    if "zarr:metadata" in src:
        # TODO: handle zarr:chunks for reference filesystem
        return dict(src["zarr:metadata"])

    if "zarr_consolidated_format" in src:
        return dict(src["metadata"])

    if ".zmetadata" in src:
        return dict(json.loads(src[".zmetadata"])["metadata"])

    return None


def _from_zarr_spec(
    spec_doc: ZarrSpecDict,
    *,
    regen_coords: bool = False,
    chunk_store: fsspec.AbstractFileSystem | Mapping[str, Any] | None = None,
    chunks=None,
    target: str | None = None,
    fsspec_opts: dict[str, Any] | None = None,
    drop_variables: Sequence[str] = (),
) -> xr.Dataset:
    fsspec_opts = fsspec_opts or {}
    if target is not None:
        if chunk_store is None:
            fs, target = url_to_fs(target, **fsspec_opts)
            chunk_store = fs.get_mapper(target)
        elif isinstance(chunk_store, fsspec.AbstractFileSystem):
            chunk_store = chunk_store.get_mapper(target)

    # TODO: deal with coordinates being loaded at open time.
    #
    # When chunk store is supplied xarray will try to load index coords (i.e.
    # name == dim, coords)

    xx = xr.open_zarr(
        spec_doc,
        chunk_store=chunk_store,
        drop_variables=drop_variables,
        chunks=chunks,
        decode_coords="all",
        consolidated=False,
    )
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
        return Context(self.geobox, self.chunks, driver=self.driver)

    @property
    def fs(self) -> fsspec.AbstractFileSystem | None:
        if self.driver is None:
            return None
        return self.driver.fs


class XrSource:
    """
    RasterSource -> xr.DataArray|xr.Dataset
    """

    def __init__(
        self,
        src: RasterSource,
        chunks: Any | None = None,
        chunk_store: (
            fsspec.AbstractFileSystem | fsspec.FSMap | Mapping[str, Any] | None
        ) = None,
        drop_variables: Sequence[str] = (),
    ) -> None:
        if isinstance(chunk_store, fsspec.AbstractFileSystem):
            chunk_store = chunk_store.get_mapper(src.uri)

        driver_data: xr.DataArray | xr.Dataset | SomeDoc = src.driver_data
        self._spec: ZarrSpecDict | None = None
        self._ds: xr.Dataset | None = None
        self._xx: xr.DataArray | None = None
        self._src = src
        self._chunks = chunks
        self._chunk_store = chunk_store
        self._drop_variables = drop_variables

        subdataset = self._src.subdataset

        if isinstance(driver_data, xr.DataArray):
            self._xx = driver_data
        elif isinstance(driver_data, xr.Dataset):
            self._ds = driver_data
            assert subdataset is not None
            assert subdataset in driver_data.data_vars
            self._xx = driver_data.data_vars[subdataset]
        elif isinstance(driver_data, dict):
            spec = extract_zarr_spec(driver_data)
            if spec is None:
                raise ValueError(f"Unsupported driver data: {type(driver_data)}")

            # create unloadable xarray.Dataset
            ds = xr.open_zarr(spec, consolidated=False, decode_coords="all", chunks={})
            assert subdataset is not None
            assert subdataset in ds.data_vars

            if chunk_store is None:
                chunk_store = fsspec.get_mapper(src.uri)

            # recreate xr.DataArray with all the dims/coords/attrs
            # but this time loadable from chunk_store
            xx = ds.data_vars[subdataset]
            xx = xr.DataArray(
                da.from_zarr(
                    spec,
                    component=subdataset,
                    chunk_store=chunk_store,
                ),
                coords=xx.coords,
                dims=xx.dims,
                name=xx.name,
                attrs=xx.attrs,
            )
            assert xx.odc.geobox is not None
            self._spec = spec
            self._ds = ds
            self._xx = xx

        elif driver_data is not None:
            raise ValueError(f"Unsupported driver data type: {type(driver_data)}")

    @property
    def spec(self) -> ZarrSpecDict | None:
        return self._spec

    @property
    def geobox(self) -> GeoBoxBase | None:
        if self._src.geobox is not None:
            return self._src.geobox
        return self.resolve().odc.geobox

    def base(
        self,
        regen_coords: bool = False,
        refresh: bool = False,
    ) -> xr.Dataset | None:
        if refresh and self._spec:
            self._ds = None

        if self._ds is not None:
            return self._ds
        if self._spec is None:
            return None
        self._ds = _from_zarr_spec(
            self._spec,
            regen_coords=regen_coords,
            chunk_store=self._chunk_store,
            target=self._src.uri,
            chunks=self._chunks,
        )
        return self._ds

    def resolve(
        self,
        regen_coords: bool = False,
        refresh: bool = False,
    ) -> xr.DataArray:
        if refresh:
            self._xx = None

        if self._xx is not None:
            return self._xx

        src_ds = self.base(regen_coords=regen_coords, refresh=refresh)
        if src_ds is None:
            raise ValueError("Failed to interpret driver data")

        subdataset = self._src.subdataset
        assert subdataset is not None

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
        self._src = XrSource(src, chunks=None, chunk_store=ctx.fs)

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
        if is_dask_collection(warped):
            warped = warped.data.compute(scheduler="synchronous")
        else:
            warped = warped.data

        assert isinstance(warped, np.ndarray)

        if dst is None:
            dst = warped
        else:
            dst[...] = warped

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
        src: xr.DataArray | None = None,
        layer_name: str = "",
        idx: int = -1,
    ) -> None:
        self._layer_name = layer_name
        self._idx = idx
        self._xx = src

    def read(
        self,
        cfg: RasterLoadParams,
        dst_geobox: GeoBox,
        *,
        selection: ReaderSubsetSelection | None = None,
        idx: tuple[int, ...] = (),
    ) -> Delayed:
        assert self._xx is not None
        assert isinstance(idx, tuple)

        xx = _subset_src(self._xx, selection, cfg)
        assert xx.odc.geobox is not None
        assert not math.isnan(xx.odc.geobox.transform.a)

        yy = xr_reproject(
            xx,
            dst_geobox,
            resampling=cfg.resampling,
            dst_nodata=cfg.fill_value,
            dtype=cfg.dtype,
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
        assert ctx is not None
        _src = XrSource(src, chunks={}, chunk_store=ctx.fs)
        xx = _src.resolve(regen_coords=True)
        assert xx.odc.geobox is not None
        assert not any(map(math.isnan, xx.odc.geobox.transform[:6]))
        return XrMemReaderDask(xx, layer_name=layer_name, idx=idx)


class XrMemReaderDriver:
    """
    Read from in memory xarray.Dataset or zarr spec document.
    """

    Reader = XrMemReader

    def __init__(
        self,
        src: xr.Dataset | None = None,
        template: RasterGroupMetadata | None = None,
        fs: fsspec.AbstractFileSystem | None = None,
    ) -> None:
        if src is not None and template is None:
            template = raster_group_md(src)
        if template is None:
            template = RasterGroupMetadata({}, {}, {}, [])
        self.src = src
        self.template = template
        self.fs = fs

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


def _zarr_chunk_refs(
    zspec: SomeDoc,
    href: str,
    *,
    bands: Sequence[str] | None = None,
    sep: str = ".",
    overrides: dict[str, Any] | None = None,
) -> Iterator[tuple[str, Any]]:
    if ".zmetadata" in zspec:
        zspec = json.loads(zspec[".zmetadata"])["metadata"]
    elif "zarr:metadata" in zspec:
        zspec = zspec["zarr:metadata"]

    assert ".zgroup" in zspec, "Not a zarr spec"

    href = href.rstrip("/")

    if bands is None:
        _bands = [k.rsplit("/", 1)[0] for k in zspec if k.endswith("/.zarray")]
    else:
        _bands = list(bands)

    if overrides is None:
        overrides = {}

    for b in _bands:
        meta = zspec[f"{b}/.zarray"]
        assert "chunks" in meta and "shape" in meta

        shape_in_blocks = tuple(
            map(len, normalize_chunks(meta["chunks"], shape=meta["shape"]))
        )

        for idx in np.ndindex(shape_in_blocks):
            if idx == ():
                k = f"{b}/0"
            else:
                k = f"{b}/{sep.join(map(str, idx))}"
            v = overrides.get(k, None)
            if v is None:
                v = (f"{href}/{k}",)

            yield (k, v)
