"""stac.load - dc.load from STAC Items."""

from __future__ import annotations

import dataclasses
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import (
    Any,
    Dict,
    Hashable,
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
from dask.array.core import normalize_chunks
from dask.base import quote, tokenize
from numpy.typing import DTypeLike
from odc.geo.geobox import GeoBox, GeoBoxBase, GeoboxTiles
from odc.geo.xr import xr_coords

from ._dask import unpack_chunks
from ._reader import nodata_mask, resolve_src_nodata
from ._utils import SizedIterable, pmap
from .types import (
    FixedCoord,
    MultiBandRasterSource,
    RasterGroupMetadata,
    RasterLoadParams,
    RasterReader,
    RasterSource,
    ReaderDriver,
)


class MkArray(Protocol):
    """Internal interface."""

    # pylint: disable=too-few-public-methods
    def __call__(
        self,
        shape: Tuple[int, ...],
        dtype: DTypeLike,
        /,
        name: Hashable,
    ) -> Any: ...  # pragma: no cover


@dataclasses.dataclass(frozen=True)
class LoadChunkTask:
    """
    Unit of work for dask graph builder.
    """

    band: str
    srcs: List[Tuple[int, str]]
    cfg: RasterLoadParams
    gbt: GeoboxTiles
    idx_tyx: Tuple[int, int, int]
    postfix_dims: Tuple[int, ...] = ()

    @property
    def dst_roi(self):
        t, y, x = self.idx_tyx
        return (t, *self.gbt.roi[y, x]) + tuple([slice(None)] * len(self.postfix_dims))

    @property
    def dst_gbox(self) -> GeoBox:
        _, y, x = self.idx_tyx
        return cast(GeoBox, self.gbt[y, x])


class DaskGraphBuilder:
    """
    Build xarray from parsed metadata.
    """

    # pylint: disable=too-few-public-methods,too-many-instance-attributes

    def __init__(
        self,
        cfg: Dict[str, RasterLoadParams],
        template: RasterGroupMetadata,
        srcs: Sequence[MultiBandRasterSource],
        tyx_bins: Dict[Tuple[int, int, int], List[int]],
        gbt: GeoboxTiles,
        env: Dict[str, Any],
        rdr: ReaderDriver,
        time_chunks: int = 1,
    ) -> None:
        self.cfg = cfg
        self.template = template
        self.srcs = srcs
        self.tyx_bins = tyx_bins
        self.gbt = gbt
        self.env = env
        self.rdr = rdr
        self._tk = tokenize(srcs, cfg, gbt, tyx_bins, env, time_chunks)
        self.chunk_shape = (time_chunks, *self.gbt.chunk_shape((0, 0)).yx)
        self._load_state = rdr.new_load(dict(zip(["time", "y", "x"], self.chunk_shape)))

    def build(
        self,
        gbox: GeoBox,
        time: Sequence[datetime],
        bands: Dict[str, RasterLoadParams],
    ):
        return mk_dataset(
            gbox,
            time,
            bands,
            self,
            extra_coords=self.template.extra_coords,
            extra_dims=self.template.extra_dims,
        )

    def __call__(
        self,
        shape: Tuple[int, ...],
        dtype: DTypeLike,
        /,
        name: Hashable,
    ) -> Any:
        # pylint: disable=too-many-locals
        assert isinstance(name, str)
        cfg = self.cfg[name]
        assert dtype == cfg.dtype
        # TODO: assumes postfix dims only for now
        ydim = 1
        post_fix_dims = shape[ydim + 2 :]

        chunk_shape = (*self.chunk_shape, *post_fix_dims)
        assert len(chunk_shape) == len(shape)
        chunks = unpack_chunks(chunk_shape, shape)
        tchunk_range = [
            range(last - n, last) for last, n in zip(np.cumsum(chunks[0]), chunks[0])
        ]

        cfg_dask_key = f"cfg-{tokenize(cfg)}"
        gbt_dask_key = f"grid-{tokenize(self.gbt)}"

        dsk: Dict[Hashable, Any] = {
            cfg_dask_key: cfg,
            gbt_dask_key: self.gbt,
        }
        tk = self._tk
        band_key = f"{name}-{tk}"
        src_key = f"open-{name}-{tk}"
        shape_in_blocks = tuple(len(ch) for ch in chunks)

        for src_idx, src in enumerate(self.srcs):
            band = src.get(name, None)
            if band is not None:
                dsk[src_key, src_idx] = (
                    _dask_open_reader,
                    band,
                    self.rdr,
                    self.env,
                    self._load_state,
                )

        for block_idx in np.ndindex(shape_in_blocks):
            ti, yi, xi = block_idx[0], block_idx[ydim], block_idx[ydim + 1]
            srcs = []
            for _ti in tchunk_range[ti]:
                srcs.append(
                    [
                        (src_key, src_idx)
                        for src_idx in self.tyx_bins.get((_ti, yi, xi), [])
                        if (src_key, src_idx) in dsk
                    ]
                )

            dsk[(band_key, *block_idx)] = (
                _dask_loader_tyx,
                srcs,
                gbt_dask_key,
                quote((yi, xi)),
                quote(post_fix_dims),
                cfg_dask_key,
                self.rdr,
                self.env,
                self._load_state,
            )

        return da.Array(dsk, band_key, chunks, dtype=dtype, shape=shape)


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
    postfix_dims: Tuple[int, ...],
    cfg: RasterLoadParams,
    rdr: ReaderDriver,
    env: Dict[str, Any],
    load_state: Any,
):
    assert cfg.dtype is not None
    gbox = cast(GeoBox, gbt[iyx])
    chunk = np.empty((len(srcs), *gbox.shape.yx, *postfix_dims), dtype=cfg.dtype)
    with rdr.restore_env(env, load_state):
        for ti, ti_srcs in enumerate(srcs):
            _fill_nd_slice(ti_srcs, gbox, cfg, chunk[ti])
        return chunk


def _fill_nd_slice(
    srcs: Sequence[RasterReader],
    dst_gbox: GeoBox,
    cfg: RasterLoadParams,
    dst: Any,
) -> Any:
    # TODO: support masks not just nodata based fusing
    #
    # ``nodata``     marks missing pixels, but it might be None (everything is valid)
    # ``fill_value`` is the initial value to use, it's equal to ``nodata`` when set,
    #                otherwise defaults to .nan for floats and 0 for integers

    # assume dst[y, x, ...] axis order
    assert dst.shape[:2] == dst_gbox.shape.yx
    postfix_roi = (slice(None),) * len(dst.shape[2:])

    nodata = resolve_src_nodata(cfg.fill_value, cfg)

    if nodata is None:
        fill_value = float("nan") if dst.dtype.kind == "f" else 0
    else:
        fill_value = nodata

    np.copyto(dst, fill_value)
    if len(srcs) == 0:
        return dst

    src, *rest = srcs
    yx_roi, pix = src.read(cfg, dst_gbox, dst=dst)
    assert len(yx_roi) == 2
    assert pix.ndim == dst.ndim

    for src in rest:
        # first valid pixel takes precedence over others
        yx_roi, pix = src.read(cfg, dst_gbox)
        assert len(yx_roi) == 2
        assert pix.ndim == dst.ndim

        _roi: Tuple[slice,] = yx_roi + postfix_roi  # type: ignore
        assert dst[_roi].shape == pix.shape

        # nodata mask takes care of nan when working with floats
        # so you can still get proper mask even when nodata is None
        # when working with float32 data.
        missing = nodata_mask(dst[_roi], nodata)
        np.copyto(dst[_roi], pix, where=missing)

    return dst


def mk_dataset(
    gbox: GeoBox,
    time: Sequence[datetime],
    bands: Dict[str, RasterLoadParams],
    alloc: Optional[MkArray] = None,
    *,
    extra_coords: Sequence[FixedCoord] | None = None,
    extra_dims: Mapping[str, int] | None = None,
) -> xr.Dataset:
    coords = xr_coords(gbox)
    crs_coord_name: Hashable = list(coords)[-1]
    coords["time"] = xr.DataArray(time, dims=("time",))
    _coords: Mapping[str, xr.DataArray] = {}
    _dims: Dict[str, int] = {}

    if extra_coords is not None:
        _coords = {
            coord.name: xr.DataArray(
                np.array(coord.values, dtype=coord.dtype),
                dims=(coord.name,),
                name=coord.name,
            )
            for coord in extra_coords
        }
        _dims.update({coord.name: len(coord.values) for coord in extra_coords})

    if extra_dims is not None:
        _dims.update(extra_dims)

    def _alloc(shape: Tuple[int, ...], dtype: str, name: Hashable) -> Any:
        if alloc is not None:
            return alloc(shape, dtype, name=name)
        return np.empty(shape, dtype=dtype)

    def _maker(name: Hashable, band: RasterLoadParams) -> xr.DataArray:
        assert band.dtype is not None
        band_coords = {**coords}

        if band.dims is not None and len(band.dims) > 2:
            # TODO: generalize to more dims
            ydim = 0
            postfix_dims = band.dims[ydim + 2 :]
            assert band.dims[ydim : ydim + 2] == ("y", "x")

            dims: Tuple[str, ...] = ("time", *gbox.dimensions, *postfix_dims)
            shape: Tuple[int, ...] = (
                len(time),
                *gbox.shape.yx,
                *[_dims[dim] for dim in postfix_dims],
            )

            band_coords.update(
                {
                    _coords[dim].name: _coords[dim]
                    for dim in postfix_dims
                    if dim in _coords
                }
            )
        else:
            dims = ("time", *gbox.dimensions)
            shape = (len(time), *gbox.shape.yx)

        data = _alloc(shape, band.dtype, name=name)
        attrs = {}
        if band.fill_value is not None:
            attrs["nodata"] = band.fill_value

        xx = xr.DataArray(data=data, coords=band_coords, dims=dims, attrs=attrs)
        xx.encoding.update(grid_mapping=crs_coord_name)
        return xx

    return xr.Dataset({name: _maker(name, band) for name, band in bands.items()})


def chunked_load(
    load_cfg: Dict[str, RasterLoadParams],
    template: RasterGroupMetadata,
    srcs: Sequence[MultiBandRasterSource],
    tyx_bins: Dict[Tuple[int, int, int], List[int]],
    gbt: GeoboxTiles,
    tss: Sequence[datetime],
    env: Dict[str, Any],
    rdr: ReaderDriver,
    *,
    chunks: Dict[str, int | Literal["auto"]] | None = None,
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
    load_cfg: Dict[str, RasterLoadParams],
    template: RasterGroupMetadata,
    srcs: Sequence[MultiBandRasterSource],
    tyx_bins: Dict[Tuple[int, int, int], List[int]],
    gbt: GeoboxTiles,
    tss: Sequence[datetime],
    env: Dict[str, Any],
    rdr: ReaderDriver,
    *,
    chunks: Dict[str, int | Literal["auto"]] | None = None,
) -> xr.Dataset:
    """Builds Dask graph for data loading."""
    if chunks is None:
        chunks = {}

    gbox = gbt.base
    chunk_shape = resolve_chunk_shape(len(tss), gbox, chunks)
    dask_loader = DaskGraphBuilder(
        load_cfg,
        template,
        srcs,
        tyx_bins,
        gbt,
        env,
        rdr,
        time_chunks=chunk_shape[0],
    )
    assert isinstance(gbox, GeoBox)
    return dask_loader.build(gbox, tss, load_cfg)


def direct_chunked_load(
    load_cfg: Dict[str, RasterLoadParams],
    template: RasterGroupMetadata,
    srcs: Sequence[MultiBandRasterSource],
    tyx_bins: Dict[Tuple[int, int, int], List[int]],
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
    bands = list(load_cfg)
    gbox = gbt.base
    assert isinstance(gbox, GeoBox)
    ds = mk_dataset(
        gbox,
        tss,
        load_cfg,
        extra_coords=template.extra_coords,
        extra_dims=template.extra_dims,
    )
    ny, nx = gbt.shape.yx
    total_tasks = nt * nb * ny * nx
    load_state = rdr.new_load()

    def _task_stream(bands: List[str]) -> Iterator[LoadChunkTask]:
        _shape: Tuple[int, int, int] = (nt, *gbt.shape.yx)
        for band_name in bands:
            cfg = load_cfg[band_name]
            for ti, yi, xi in np.ndindex(_shape):  # type: ignore
                tyx_idx = (ti, yi, xi)
                _srcs = [(idx, band_name) for idx in tyx_bins.get(tyx_idx, [])]
                yield LoadChunkTask(band_name, _srcs, cfg, gbt, tyx_idx)

    def _do_one(task: LoadChunkTask) -> Tuple[str, int, int, int]:
        dst_slice = ds[task.band].data[task.dst_roi]
        _srcs = [
            src
            for src in (srcs[idx].get(band, None) for idx, band in task.srcs)
            if src is not None
        ]
        with rdr.restore_env(env, load_state) as ctx:
            loaders = [rdr.open(src, ctx) for src in _srcs]
            _ = _fill_nd_slice(
                loaders,
                task.dst_gbox,
                task.cfg,
                dst=dst_slice,
            )
        t, y, x = task.idx_tyx
        return (task.band, t, y, x)

    _work = pmap(_do_one, _task_stream(bands), pool)

    if progress is not None:
        _work = progress(SizedIterable(_work, total_tasks))

    for _ in _work:
        pass

    rdr.finalise_load(load_state)
    return ds


def resolve_chunk_shape(
    nt: int,
    gbox: GeoBoxBase,
    chunks: Dict[str, int | Literal["auto"]],
    dtype: Any | None = None,
    cfg: Mapping[str, RasterLoadParams] | None = None,
) -> Tuple[int, int, int]:
    """
    Compute chunk size for time, y and x dimensions.
    """
    if cfg is None:
        cfg = {}

    if dtype is None:
        _dtypes = sorted(
            set(cfg.dtype for cfg in cfg.values() if cfg.dtype is not None),
            key=lambda x: np.dtype(x).itemsize,
            reverse=True,
        )
        dtype = "uint16" if len(_dtypes) == 0 else _dtypes[0]

    tt = chunks.get("time", 1)
    ty, tx = (
        chunks.get(dim, chunks.get(fallback_dim, -1))
        for dim, fallback_dim in zip(gbox.dimensions, ["y", "x"])
    )
    nt, ny, nx = (
        ch[0]
        for ch in normalize_chunks((tt, ty, tx), (nt, *gbox.shape.yx), dtype=dtype)
    )

    return nt, ny, nx
