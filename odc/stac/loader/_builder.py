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
from odc.geo.geobox import GeoBox, GeoboxTiles
from odc.geo.xr import xr_coords

from ._dask import unpack_chunks
from ._reader import nodata_mask, resolve_src_nodata
from ._utils import SizedIterable, pmap
from .types import MultiBandRasterSource, RasterLoadParams, RasterSource, SomeReader


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

    @property
    def dst_roi(self):
        t, y, x = self.idx_tyx
        return (t, *self.gbt.roi[y, x])

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
        srcs: Sequence[MultiBandRasterSource],
        tyx_bins: Dict[Tuple[int, int, int], List[int]],
        gbt: GeoboxTiles,
        env: Dict[str, Any],
        rdr: SomeReader,
        time_chunks: int = 1,
    ) -> None:
        self.cfg = cfg
        self.srcs = srcs
        self.tyx_bins = tyx_bins
        self.gbt = gbt
        self.env = env
        self.rdr = rdr
        self._tk = tokenize(srcs, cfg, gbt, tyx_bins, env, time_chunks)
        self.chunk_shape = (time_chunks, *self.gbt.chunk_shape((0, 0)).yx)

    def __call__(
        self,
        shape: Tuple[int, ...],
        dtype: DTypeLike,
        /,
        name: Hashable,
    ) -> Any:
        # pylint: disable=too-many-locals
        assert len(shape) == 3
        assert isinstance(name, str)
        cfg = self.cfg[name]
        assert dtype == cfg.dtype

        chunks = unpack_chunks(self.chunk_shape, shape)
        tchunk_range = [
            range(last - n, last) for last, n in zip(np.cumsum(chunks[0]), chunks[0])
        ]

        cfg_key = f"cfg-{tokenize(cfg)}"
        gbt_key = f"grid-{tokenize(self.gbt)}"

        dsk: Dict[Hashable, Any] = {
            cfg_key: cfg,
            gbt_key: self.gbt,
        }
        tk = self._tk
        band_key = f"{name}-{tk}"
        md_key = f"md-{name}-{tk}"
        shape_in_blocks = tuple(len(ch) for ch in chunks)

        for idx, src in enumerate(self.srcs):
            band = src.get(name, None)
            if band is not None:
                dsk[md_key, idx] = band

        for ti, yi, xi in np.ndindex(shape_in_blocks):  # type: ignore
            srcs = []
            for _ti in tchunk_range[ti]:
                srcs.append(
                    [
                        (md_key, idx)
                        for idx in self.tyx_bins.get((_ti, yi, xi), [])
                        if (md_key, idx) in dsk
                    ]
                )

            dsk[band_key, ti, yi, xi] = (
                _dask_loader_tyx,
                srcs,
                gbt_key,
                quote((yi, xi)),
                self.rdr,
                cfg_key,
                self.env,
            )

        return da.Array(dsk, band_key, chunks, dtype=dtype, shape=shape)


def _dask_loader_tyx(
    srcs: Sequence[Sequence[RasterSource]],
    gbt: GeoboxTiles,
    iyx: Tuple[int, int],
    rdr: SomeReader,
    cfg: RasterLoadParams,
    env: Dict[str, Any],
):
    assert cfg.dtype is not None
    gbox = cast(GeoBox, gbt[iyx])
    chunk = np.empty((len(srcs), *gbox.shape.yx), dtype=cfg.dtype)
    with rdr.restore_env(env):
        for i, plane in enumerate(srcs):
            fill_2d_slice(plane, gbox, cfg, rdr, chunk[i, :, :])
        return chunk


def fill_2d_slice(
    srcs: Sequence[RasterSource],
    dst_gbox: GeoBox,
    cfg: RasterLoadParams,
    rdr: SomeReader,
    dst: Any,
) -> Any:
    # TODO: support masks not just nodata based fusing
    #
    # ``nodata``     marks missing pixels, but it might be None (everything is valid)
    # ``fill_value`` is the initial value to use, it's equal to ``nodata`` when set,
    #                otherwise defaults to .nan for floats and 0 for integers
    assert dst.shape == dst_gbox.shape.yx
    nodata = resolve_src_nodata(cfg.fill_value, cfg)

    if nodata is None:
        fill_value = float("nan") if dst.dtype.kind == "f" else 0
    else:
        fill_value = nodata

    np.copyto(dst, fill_value)
    if len(srcs) == 0:
        return dst

    src, *rest = srcs
    _roi, pix = rdr.read(src, cfg, dst_gbox, dst=dst)

    for src in rest:
        # first valid pixel takes precedence over others
        _roi, pix = rdr.read(src, cfg, dst_gbox)

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
) -> xr.Dataset:
    _shape = (len(time), *gbox.shape.yx)
    coords = xr_coords(gbox)
    crs_coord_name: Hashable = list(coords)[-1]
    coords["time"] = xr.DataArray(time, dims=("time",))
    dims = ("time", *gbox.dimensions)

    def _alloc(shape: Tuple[int, ...], dtype: str, name: Hashable) -> Any:
        if alloc is not None:
            return alloc(shape, dtype, name=name)
        return np.empty(shape, dtype=dtype)

    def _maker(name: Hashable, band: RasterLoadParams) -> xr.DataArray:
        assert band.dtype is not None
        data = _alloc(_shape, band.dtype, name=name)
        attrs = {}
        if band.fill_value is not None:
            attrs["nodata"] = band.fill_value

        xx = xr.DataArray(data=data, coords=coords, dims=dims, attrs=attrs)
        xx.encoding.update(grid_mapping=crs_coord_name)
        return xx

    return xr.Dataset({name: _maker(name, band) for name, band in bands.items()})


def direct_chunked_load(
    load_cfg: Dict[str, RasterLoadParams],
    srcs: Sequence[MultiBandRasterSource],
    tyx_bins: Dict[Tuple[int, int, int], List[int]],
    gbt: GeoboxTiles,
    tss: Sequence[datetime],
    env: Dict[str, Any],
    rdr: SomeReader,
    *,
    pool: ThreadPoolExecutor | int | None = None,
    progress: Optional[Any] = None,
) -> xr.Dataset:
    """
    Load in chunks but without using Dask.
    """
    # pylint: disable=too-many-locals
    assert len(tss) == len(srcs)
    nt = len(tss)
    nb = len(load_cfg)
    bands = list(load_cfg)
    gbox = gbt.base
    assert isinstance(gbox, GeoBox)
    ds = mk_dataset(gbox, tss, load_cfg)
    ny, nx = gbt.shape.yx
    total_tasks = nt * nb * ny * nx

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
        with rdr.restore_env(env):
            _ = fill_2d_slice(_srcs, task.dst_gbox, task.cfg, rdr, dst_slice)
        t, y, x = task.idx_tyx
        return (task.band, t, y, x)

    _work = pmap(_do_one, _task_stream(bands), pool)

    if progress is not None:
        _work = progress(SizedIterable(_work, total_tasks))

    for _ in _work:
        pass

    return ds


def resolve_chunk_shape(
    nt: int,
    gbox: GeoBox,
    chunks: Dict[str, int | Literal["auto"]],
    dtype: Any,
) -> Tuple[int, int, int]:
    """
    Compute chunk size for time, y and x dimensions.
    """
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
