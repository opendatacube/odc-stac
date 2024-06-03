"""stac.load - dc.load from STAC Items."""

from __future__ import annotations

import dataclasses
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import (
    Any,
    Dict,
    Hashable,
    Iterable,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    cast,
)

import numpy as np
import xarray as xr
from dask import array as da
from dask import is_dask_collection
from dask.array.core import normalize_chunks
from dask.base import quote, tokenize
from dask.highlevelgraph import HighLevelGraph
from dask.typing import Key
from numpy.typing import DTypeLike
from odc.geo.geobox import GeoBox, GeoBoxBase, GeoboxTiles
from odc.geo.xr import xr_coords

from ._reader import (
    ReaderDaskAdaptor,
    nodata_mask,
    resolve_dst_fill_value,
    resolve_src_nodata,
)
from ._utils import SizedIterable, pmap
from .types import (
    DaskRasterReader,
    MultiBandRasterSource,
    RasterGroupMetadata,
    RasterLoadParams,
    RasterReader,
    RasterSource,
    ReaderDriver,
    T,
)

DaskBuilderMode = Literal["mem", "concurrency"]


class MkArray(Protocol):
    """Internal interface."""

    # pylint: disable=too-few-public-methods
    def __call__(
        self,
        shape: Tuple[int, ...],
        dtype: DTypeLike,
        /,
        name: Hashable,
        ydim: int,
    ) -> Any: ...  # pragma: no cover


@dataclasses.dataclass(frozen=True)
class LoadChunkTask:
    """
    Unit of work for dask graph builder.
    """

    # pylint: disable=too-many-instance-attributes

    band: str
    srcs: List[List[int]]
    cfg: RasterLoadParams
    gbt: GeoboxTiles
    idx: Tuple[int, ...]
    shape: Tuple[int, ...]
    ydim: int = 1
    selection: Any = None  # optional slice into extra dims

    @property
    def idx_tyx(self) -> Tuple[int, int, int]:
        ydim = self.ydim
        return self.idx[0], self.idx[ydim], self.idx[ydim + 1]

    @property
    def prefix_dims(self) -> tuple[int, ...]:
        return self.shape[1 : self.ydim]

    @property
    def postfix_dims(self) -> tuple[int, ...]:
        return self.shape[self.ydim + 2 :]

    @property
    def dst_roi(self) -> Tuple[slice, ...]:
        t, y, x = self.idx_tyx
        iy, ix = self.gbt.roi[y, x]
        return (
            slice(t, t + len(self.srcs)),
            *[slice(None) for _ in self.prefix_dims],
            iy,
            ix,
            *[slice(None) for _ in self.postfix_dims],
        )

    @property
    def dst_gbox(self) -> GeoBox:
        _, y, x = self.idx_tyx
        return cast(GeoBox, self.gbt[y, x])

    def __bool__(self) -> bool:
        return len(self.srcs) > 0 and any(len(src) > 0 for src in self.srcs)

    def resolve_sources(
        self, srcs: Sequence[MultiBandRasterSource]
    ) -> List[List[RasterSource]]:
        out: List[List[RasterSource]] = []

        for layer in self.srcs:
            _srcs: List[RasterSource] = []
            for idx in layer:
                src = srcs[idx].get(self.band, None)
                if src is not None:
                    _srcs.append(src)
            out.append(_srcs)
        return out

    def resolve_sources_dask(
        self, dask_key: str, dsk: Mapping[Key, Any] | None = None
    ) -> list[list[tuple[str, int]]]:
        if dsk is None:
            return [[(dask_key, idx) for idx in layer] for layer in self.srcs]

        # Skip missing sources
        return [
            [(dask_key, idx) for idx in layer if (dask_key, idx) in dsk]
            for layer in self.srcs
        ]


def _default_dask_mode() -> DaskBuilderMode:
    mode = os.environ.get("ODC_STAC_DASK_MODE", "mem")
    if mode == "concurrency":
        return "concurrency"
    return "mem"


class DaskGraphBuilder:
    """
    Build xarray from parsed metadata.
    """

    # pylint: disable=too-few-public-methods,too-many-instance-attributes

    def __init__(
        self,
        cfg: Mapping[str, RasterLoadParams],
        template: RasterGroupMetadata,
        srcs: Sequence[MultiBandRasterSource],
        tyx_bins: Mapping[Tuple[int, int, int], List[int]],
        gbt: GeoboxTiles,
        env: Dict[str, Any],
        rdr: ReaderDriver,
        chunks: Mapping[str, int],
        mode: DaskBuilderMode | Literal["auto"] = "auto",
    ) -> None:
        gbox = gbt.base
        assert isinstance(gbox, GeoBox)
        # make sure chunks for tyx match our structure
        chunk_tyx = (chunks.get("time", 1), *gbt.chunk_shape((0, 0)).yx)
        chunks = {**chunks}
        chunks.update(dict(zip(["time", "y", "x"], chunk_tyx)))
        if mode == "auto":
            # "mem" unless overwriten by env var
            mode = _default_dask_mode()

        self.cfg = cfg
        self.template = template
        self.srcs = srcs
        self.tyx_bins = tyx_bins
        self.gbt = gbt
        self.env = env
        self.rdr = rdr
        self._tk = tokenize(srcs, cfg, gbt, tyx_bins, env, chunks, mode)
        self._chunks = chunks
        self._mode = mode
        self._load_state = rdr.new_load(gbox, chunks=chunks)

    def _band_chunks(
        self,
        band: str,
        shape: tuple[int, ...],
        ydim: int,
    ) -> tuple[tuple[int, ...], ...]:
        chunks = resolve_chunks(
            (shape[0], shape[ydim], shape[ydim + 1]),
            self._chunks,
            extra_dims=self.template.extra_dims_full(band),
        )
        return denorm_ydim(chunks, ydim)

    def build(
        self,
        gbox: GeoBox,
        time: Sequence[datetime],
        bands: Mapping[str, RasterLoadParams],
    ) -> xr.Dataset:
        return mk_dataset(
            gbox,
            time,
            bands,
            self,
            template=self.template,
        )

    def _norm_load_state(self, cfg_layer: dict[Key, Any]) -> tuple[Any, Any]:
        load_state = self._load_state
        if is_dask_collection(load_state):
            cfg_layer.update(load_state.dask)
            return load_state, load_state.key

        return load_state, load_state

    def _prep_sources(
        self,
        name: str,
        dsk: dict[Key, Any],
        load_state_dsk: Any,
    ) -> dict[Key, Any]:
        src_key = f"open-{name}-{self._tk}"

        for src_idx, mbsrc in enumerate(self.srcs):
            rsrc = mbsrc.get(name, None)
            if rsrc is not None:
                dsk[src_key, src_idx] = (
                    _dask_open_reader,
                    rsrc,
                    self.rdr,
                    self.env,
                    load_state_dsk,
                )
        return dsk

    def _dask_rdr(self) -> DaskRasterReader:
        if (dask_reader := self.rdr.dask_reader) is not None:
            return dask_reader
        return ReaderDaskAdaptor(self.rdr, self.env)

    def _task_futures(
        self,
        task: LoadChunkTask,
        dask_reader: DaskRasterReader,
        layer_name: str,
        dsk: dict[Key, Any],
    ) -> list[list[Key]]:
        # pylint: disable=too-many-locals
        srcs = task.resolve_sources(self.srcs)
        out: list[list[Key]] = []
        ctx = self._load_state
        cfg = task.cfg
        dst_gbox = task.dst_gbox

        for i_time, layer in enumerate(srcs, start=task.idx[0]):
            keys_out: list[Key] = []
            for i_src, src in enumerate(layer):
                idx = (i_time, *task.idx[1:], i_src)
                rdr = dask_reader.open(src, ctx, layer_name=layer_name)
                fut = rdr.read(cfg, dst_gbox, selection=task.selection, idx=idx)
                keys_out.append(fut.key)
                dsk.update(fut.dask)

            out.append(keys_out)

        return out

    def __call__(
        self,
        shape: Tuple[int, ...],
        dtype: DTypeLike,
        /,
        name: Hashable,
        ydim: int,
    ) -> Any:
        # pylint: disable=too-many-locals
        assert isinstance(name, str)

        cfg = self.cfg[name]
        assert dtype == cfg.dtype
        assert ydim == cfg.ydim + 1  # +1 for time dimension
        chunks = self._band_chunks(name, shape, ydim)

        tk = self._tk
        cfg_dsk = f"cfg-{tokenize(cfg)}"
        gbt_dsk = f"grid-{tokenize(self.gbt)}"

        cfg_layer, open_layer, band_layer = (
            f"cfg-{name}-{tk}",
            f"open-{name}-{tk}",
            f"{name}-{tk}",
        )

        layers: Dict[str, Dict[Key, Any]] = {
            cfg_layer: {
                cfg_dsk: cfg,
                gbt_dsk: self.gbt,
            },
            open_layer: {},
            band_layer: {},
        }
        layer_deps: Dict[str, Any] = {
            cfg_layer: set(),
            open_layer: set([cfg_layer]),
            band_layer: set([cfg_layer, open_layer]),
        }

        dsk = layers[f"{name}-{tk}"]

        dask_reader: DaskRasterReader | None = None
        load_state, load_state_dsk = self._norm_load_state(layers[cfg_layer])
        assert load_state is load_state_dsk or is_dask_collection(load_state)

        if self._mode == "mem":
            self._prep_sources(name, layers[open_layer], load_state_dsk)
        else:
            dask_reader = self._dask_rdr()

        fill_value = resolve_dst_fill_value(
            np.dtype(dtype),
            cfg,
            resolve_src_nodata(cfg.fill_value, cfg),
        )

        for task in self.load_tasks(name, shape[0]):
            task_key: Key = (band_layer, *task.idx)
            if dask_reader is None:
                dsk[task_key] = (
                    _dask_loader_tyx,
                    task.resolve_sources_dask(open_layer, layers[open_layer]),
                    gbt_dsk,
                    quote(task.idx_tyx[1:]),
                    quote(task.prefix_dims),
                    quote(task.postfix_dims),
                    cfg_dsk,
                    self.rdr,
                    self.env,
                    load_state_dsk,
                    task.selection,
                )
            else:
                srcs_futures = self._task_futures(
                    task, dask_reader, open_layer, layers[open_layer]
                )

                dsk[task_key] = (
                    _dask_fuser,
                    srcs_futures,
                    task.shape,
                    dtype,
                    fill_value,
                    ydim - 1,
                )

        dsk = HighLevelGraph(layers, layer_deps)
        return da.Array(dsk, band_layer, chunks, dtype=dtype, shape=shape)

    def load_tasks(self, name: str, nt: int) -> Iterator[LoadChunkTask]:
        return load_tasks(
            self.cfg,
            self.tyx_bins,
            self.gbt,
            nt=nt,
            chunks=self._chunks,
            extra_dims=self.template.extra_dims_full(name),
            bands=[name],
        )


def _dask_open_reader(
    src: RasterSource,
    rdr: ReaderDriver,
    env: Dict[str, Any],
    load_state: Any,
) -> RasterReader:
    with rdr.restore_env(env, load_state) as ctx:
        return rdr.open(src, ctx)


def _dask_loader_tyx(
    srcs: Sequence[Sequence[RasterReader]],
    gbt: GeoboxTiles,
    iyx: Tuple[int, int],
    prefix_dims: Tuple[int, ...],
    postfix_dims: Tuple[int, ...],
    cfg: RasterLoadParams,
    rdr: ReaderDriver,
    env: Dict[str, Any],
    load_state: Any,
    selection: Any | None = None,
):
    assert cfg.dtype is not None
    gbox = cast(GeoBox, gbt[iyx])
    chunk = np.empty(
        (len(srcs), *prefix_dims, *gbox.shape.yx, *postfix_dims),
        dtype=cfg.dtype,
    )
    ydim = len(prefix_dims)
    with rdr.restore_env(env, load_state):
        for ti, ti_srcs in enumerate(srcs):
            _fill_nd_slice(
                ti_srcs, gbox, cfg, chunk[ti], ydim=ydim, selection=selection
            )
        return chunk


def _dask_fuser(
    srcs: list[list[Any]],
    shape: tuple[int, ...],
    dtype: DTypeLike,
    fill_value: float | int,
    src_ydim: int = 0,
):
    assert shape[0] == len(srcs)
    assert len(shape) >= 3  # time, ..., y, x, ...

    dst = np.full(shape, fill_value, dtype=dtype)

    for ti, layer in enumerate(srcs):
        fuse_nd_slices(
            layer,
            fill_value,
            dst[ti],
            ydim=src_ydim,
            prefilled=True,
        )

    return dst


def fuse_nd_slices(
    srcs: Iterable[tuple[tuple[slice, slice], np.ndarray]],
    fill_value: float | int,
    dst: Any,
    ydim: int = 0,
    prefilled: bool = False,
) -> Any:
    postfix_roi = (slice(None),) * len(dst.shape[ydim + 2 :])
    prefix_roi = (slice(None),) * ydim

    if not prefilled:
        np.copyto(dst, fill_value)

    for yx_roi, pix in srcs:
        _roi: tuple[slice, ...] = prefix_roi + yx_roi + postfix_roi  # type: ignore
        assert dst[_roi].shape == pix.shape

        missing = nodata_mask(dst[_roi], fill_value)
        np.copyto(dst[_roi], pix, where=missing)

    return dst


def _fill_nd_slice(
    srcs: Sequence[RasterReader],
    dst_gbox: GeoBox,
    cfg: RasterLoadParams,
    dst: Any,
    ydim: int = 0,
    selection: Any | None = None,
) -> Any:
    # TODO: support masks not just nodata based fusing
    #
    # ``nodata``     marks missing pixels, but it might be None (everything is valid)
    # ``fill_value`` is the initial value to use, it's equal to ``nodata`` when set,
    #                otherwise defaults to .nan for floats and 0 for integers
    # pylint: disable=too-many-locals

    assert dst.shape[ydim : ydim + 2] == dst_gbox.shape.yx
    nodata = resolve_src_nodata(cfg.fill_value, cfg)
    fill_value = resolve_dst_fill_value(dst.dtype, cfg, nodata)

    np.copyto(dst, fill_value)
    if len(srcs) == 0:
        return dst

    src, *rest = srcs
    yx_roi, pix = src.read(cfg, dst_gbox, dst=dst, selection=selection)
    assert len(yx_roi) == 2
    assert pix.ndim == dst.ndim

    return fuse_nd_slices(
        (src.read(cfg, dst_gbox, selection=selection) for src in rest),
        fill_value,
        dst,
        ydim=ydim,
        prefilled=True,
    )


def mk_dataset(
    gbox: GeoBox,
    time: Sequence[datetime],
    bands: Mapping[str, RasterLoadParams],
    alloc: Optional[MkArray] = None,
    *,
    template: RasterGroupMetadata,
) -> xr.Dataset:
    coords = xr_coords(gbox)
    crs_coord_name: Hashable = list(coords)[-1]
    coords["time"] = xr.DataArray(time, dims=("time",))
    _dims = template.extra_dims_full()

    _coords = {
        coord.name: xr.DataArray(
            np.array(coord.values, dtype=coord.dtype),
            dims=(coord.dim,),
            name=coord.name,
        )
        for coord in template.extra_coords
    }

    def _alloc(shape: Tuple[int, ...], dtype: str, name: Hashable, ydim: int) -> Any:
        if alloc is not None:
            return alloc(shape, dtype, name=name, ydim=ydim)
        return np.empty(shape, dtype=dtype)

    def _maker(name: Hashable, band: RasterLoadParams) -> xr.DataArray:
        assert band.dtype is not None
        band_coords = {**coords}
        ydim = band.ydim

        if len(band.dims) > 2:
            assert band.dims[ydim : ydim + 2] == ("y", "x")
            prefix_dims = band.dims[:ydim]
            postfix_dims = band.dims[ydim + 2 :]

            dims: Tuple[str, ...] = (
                "time",
                *prefix_dims,
                *gbox.dimensions,
                *postfix_dims,
            )
            shape: Tuple[int, ...] = (
                len(time),
                *[_dims[dim] for dim in prefix_dims],
                *gbox.shape.yx,
                *[_dims[dim] for dim in postfix_dims],
            )

            band_coords.update(
                {
                    _coords[dim].name: _coords[dim]
                    for dim in (prefix_dims + postfix_dims)
                    if dim in _coords
                }
            )
        else:
            dims = ("time", *gbox.dimensions)
            shape = (len(time), *gbox.shape.yx)

        data = _alloc(
            shape,
            band.dtype,
            name=name,
            ydim=ydim + 1,  # +1 for time dimension
        )
        attrs = {}
        if band.fill_value is not None:
            attrs["nodata"] = band.fill_value

        xx = xr.DataArray(data=data, coords=band_coords, dims=dims, attrs=attrs)
        xx.encoding.update(grid_mapping=crs_coord_name)
        return xx

    return xr.Dataset({name: _maker(name, band) for name, band in bands.items()})


def chunked_load(
    load_cfg: Mapping[str, RasterLoadParams],
    template: RasterGroupMetadata,
    srcs: Sequence[MultiBandRasterSource],
    tyx_bins: Mapping[Tuple[int, int, int], List[int]],
    gbt: GeoboxTiles,
    tss: Sequence[datetime],
    env: Dict[str, Any],
    rdr: ReaderDriver,
    *,
    chunks: Mapping[str, int | Literal["auto"]] | None = None,
    pool: ThreadPoolExecutor | int | None = None,
    progress: Optional[Any] = None,
) -> xr.Dataset:
    """
    Route to either direct or dask chunked load.
    """
    # pylint: disable=too-many-arguments
    if chunks is None:
        return direct_chunked_load(
            load_cfg,
            template,
            srcs,
            tyx_bins,
            gbt,
            tss,
            env,
            rdr,
            pool=pool,
            progress=progress,
        )
    return dask_chunked_load(
        load_cfg,
        template,
        srcs,
        tyx_bins,
        gbt,
        tss,
        env,
        rdr,
        chunks=chunks,
    )


def dask_chunked_load(
    load_cfg: Mapping[str, RasterLoadParams],
    template: RasterGroupMetadata,
    srcs: Sequence[MultiBandRasterSource],
    tyx_bins: Mapping[Tuple[int, int, int], List[int]],
    gbt: GeoboxTiles,
    tss: Sequence[datetime],
    env: Dict[str, Any],
    rdr: ReaderDriver,
    *,
    chunks: Mapping[str, int | Literal["auto"]] | None = None,
) -> xr.Dataset:
    """Builds Dask graph for data loading."""
    if chunks is None:
        chunks = {}

    gbox = gbt.base
    extra_dims = template.extra_dims_full()
    chunk_shape = resolve_chunk_shape(
        len(tss),
        gbox,
        chunks,
        extra_dims=extra_dims,
    )
    chunks_normalized = dict(zip(["time", "y", "x", *extra_dims], chunk_shape))
    dask_loader = DaskGraphBuilder(
        load_cfg,
        template,
        srcs,
        tyx_bins,
        gbt,
        env,
        rdr,
        chunks=chunks_normalized,
    )
    assert isinstance(gbox, GeoBox)
    return dask_loader.build(gbox, tss, load_cfg)


def denorm_ydim(x: tuple[T, ...], ydim: int) -> tuple[T, ...]:
    ydim = ydim - 1
    if ydim == 0:
        return x
    t, y, x, *rest = x
    return (t, *rest[:ydim], y, x, *rest[ydim:])


def load_tasks(
    load_cfg: Mapping[str, RasterLoadParams],
    tyx_bins: Mapping[Tuple[int, int, int], List[int]],
    gbt: GeoboxTiles,
    *,
    nt: Optional[int] = None,
    chunks: Mapping[str, int] | None = None,
    extra_dims: Mapping[str, int] | None = None,
    bands: Sequence[str] | None = None,
) -> Iterator[LoadChunkTask]:
    """
    Convert tyx_bins into a complete set of load tasks.

    This is a generator that yields :py:class:`~odc.loader.LoadChunkTask`
    instances for every possible time, y, x, bins, including empty ones.
    """
    # pylint: disable=too-many-locals
    extra_dims = extra_dims or {}
    chunks = chunks or {}

    if nt is None:
        nt = max(t for t, _, _ in tyx_bins) + 1

    chunks = {**chunks}
    chunks.update(zip(["y", "x"], gbt.chunk_shape((0, 0)).yx))
    base_shape = (nt, *gbt.base.shape.yx)

    if bands is None:
        bands = list(load_cfg)

    for band_name in bands:
        cfg = load_cfg[band_name]
        _edims: Mapping[str, int] = {}

        if _dims := cfg.extra_dims:
            _edims = dict((k, v) for k, v in extra_dims.items() if k in _dims)

        _chunks = resolve_chunks(base_shape, chunks, dtype=cfg.dtype, extra_dims=_edims)
        _offsets: list[tuple[int, ...]] = [
            (0, *np.cumsum(ch, dtype="int64").tolist()) for ch in _chunks
        ]
        shape_in_chunks = tuple(len(ch) for ch in _chunks)  # T,Y,X[,B]
        ndim = len(shape_in_chunks)
        ydim = cfg.ydim + 1

        for idx in np.ndindex(shape_in_chunks[:3]):
            tBi, yi, xi = idx  # type: ignore
            srcs: List[List[int]] = []
            t0, nt = _offsets[0][tBi], _chunks[0][tBi]
            for ti in range(t0, t0 + nt):
                tyx_idx = (ti, yi, xi)
                srcs.append(tyx_bins.get(tyx_idx, []))

            chunk_shape_tyx: tuple[int, ...] = tuple(
                _chunks[dim][i_chunk] for dim, i_chunk in enumerate(idx)
            )

            if ndim == 3:
                yield LoadChunkTask(
                    band_name,
                    srcs,
                    cfg,
                    gbt,
                    idx,
                    chunk_shape_tyx,
                )
                continue

            for extra_idx in np.ndindex(shape_in_chunks[3:]):
                extra_chunk_shape = tuple(
                    _chunks[dim][i_chunk]
                    for dim, i_chunk in enumerate(extra_idx, start=3)
                )
                extra_chunk_offset = (
                    _offsets[dim][i_chunk]
                    for dim, i_chunk in enumerate(extra_idx, start=3)
                )
                selection: Any = tuple(
                    slice(o, o + n)
                    for o, n in zip(extra_chunk_offset, extra_chunk_shape)
                )
                if len(selection) == 1:
                    selection = selection[0]
                    if shape_in_chunks[3] == 1:
                        selection = None

                yield LoadChunkTask(
                    band_name,
                    srcs,
                    cfg,
                    gbt,
                    denorm_ydim(idx + extra_idx, ydim),
                    denorm_ydim(chunk_shape_tyx + extra_chunk_shape, ydim),
                    ydim=ydim,
                    selection=selection,
                )


def direct_chunked_load(
    load_cfg: Mapping[str, RasterLoadParams],
    template: RasterGroupMetadata,
    srcs: Sequence[MultiBandRasterSource],
    tyx_bins: Mapping[Tuple[int, int, int], List[int]],
    gbt: GeoboxTiles,
    tss: Sequence[datetime],
    env: Dict[str, Any],
    rdr: ReaderDriver,
    *,
    pool: ThreadPoolExecutor | int | None = None,
    progress: Optional[Any] = None,
) -> xr.Dataset:
    """
    Load in chunks but without using Dask.
    """
    # pylint: disable=too-many-locals
    nt = len(tss)
    nb = len(load_cfg)
    gbox = gbt.base
    assert isinstance(gbox, GeoBox)
    ds = mk_dataset(
        gbox,
        tss,
        load_cfg,
        template=template,
    )
    ny, nx = gbt.shape.yx
    total_tasks = nt * nb * ny * nx
    load_state = rdr.new_load(gbox)

    def _do_one(task: LoadChunkTask) -> Tuple[str, int, int, int]:
        dst_slice = ds[task.band].data[task.dst_roi]
        layers = task.resolve_sources(srcs)
        ydim = len(task.prefix_dims)

        with rdr.restore_env(env, load_state) as ctx:
            for t_idx, layer in enumerate(layers):
                loaders = [rdr.open(src, ctx) for src in layer]
                _ = _fill_nd_slice(
                    loaders,
                    task.dst_gbox,
                    task.cfg,
                    dst=dst_slice[t_idx],
                    ydim=ydim,
                )
        t, y, x = task.idx_tyx
        return (task.band, t, y, x)

    tasks = load_tasks(
        load_cfg,
        tyx_bins,
        gbt,
        nt=nt,
        extra_dims=template.extra_dims_full(),
    )
    tasks = list(tasks)
    assert len(tasks) == total_tasks

    _work = pmap(_do_one, tasks, pool)

    if progress is not None:
        _work = progress(SizedIterable(_work, total_tasks))

    for _ in _work:
        pass

    rdr.finalise_load(load_state)
    return ds


def _largest_dtype(
    cfg: Mapping[str, RasterLoadParams] | None,
    fallback: str | np.dtype = "float32",
) -> np.dtype:
    if isinstance(fallback, str):
        fallback = np.dtype(fallback)

    if cfg is None:
        return fallback

    _dtypes = sorted(
        set(np.dtype(cfg.dtype) for cfg in cfg.values() if cfg.dtype is not None),
        key=lambda x: x.itemsize,
        reverse=True,
    )
    if _dtypes:
        return _dtypes[0]

    return fallback


def resolve_chunks(
    base_shape: tuple[int, int, int],
    chunks: Mapping[str, int | Literal["auto"]],
    dtype: Any | None = None,
    extra_dims: Mapping[str, int] | None = None,
    limit: Any | None = None,
) -> tuple[tuple[int, ...], ...]:
    if extra_dims is None:
        extra_dims = {}
    tt = chunks.get("time", 1)
    ty, tx = (chunks.get(dim, -1) for dim in ["y", "x"])
    chunks = (tt, ty, tx) + tuple((chunks.get(dim, -1) for dim in extra_dims))
    shape = base_shape + tuple(extra_dims.values())
    return normalize_chunks(chunks, shape, dtype=dtype, limit=limit)


def resolve_chunk_shape(
    nt: int,
    gbox: GeoBoxBase,
    chunks: Mapping[str, int | Literal["auto"]],
    dtype: Any | None = None,
    cfg: Mapping[str, RasterLoadParams] | None = None,
    extra_dims: Mapping[str, int] | None = None,
) -> Tuple[int, ...]:
    """
    Compute chunk size for time, y and x dimensions and extra dims.

    Spatial dimension chunks need to be suppliead with ``y,x`` keys.

    :returns: Chunk shape in (T,Y,X, *extra_dims) order
    """
    if dtype is None and cfg:
        dtype = _largest_dtype(cfg, "float32")

    chunks = {**chunks}
    for s, d in zip(gbox.dimensions, ["y", "x"]):
        if s != d and s in chunks:
            chunks[d] = chunks[s]

    resolved_chunks = resolve_chunks(
        (nt, *gbox.shape.yx),
        chunks,
        dtype=dtype,
        extra_dims=extra_dims,
    )
    return tuple(int(ch[0]) for ch in resolved_chunks)
