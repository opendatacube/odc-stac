# pylint: disable=missing-function-docstring,missing-module-docstring,too-many-statements,too-many-locals
# pylint: disable=redefined-outer-name,unused-argument
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace as _sn
from typing import Any, Dict, Literal, Mapping, Sequence

import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox, GeoboxTiles

from . import chunked_load
from ._builder import (
    DaskGraphBuilder,
    _largest_dtype,
    load_tasks,
    mk_dataset,
    resolve_chunk_shape,
    resolve_chunks,
)
from .testing.fixtures import FakeMDPlugin, FakeReaderDriver
from .types import (
    FixedCoord,
    RasterBandMetadata,
    RasterGroupMetadata,
    RasterLoadParams,
    RasterSource,
)

tss = [datetime(2020, 1, 1)]
gbox = GeoBox.from_bbox((-180, -90, 180, 90), shape=(160, 320), tight=True)
gbt = GeoboxTiles(gbox, (80, 80))
shape = (len(tss), *gbox.shape.yx)
dims = ("time", *gbox.dimensions)
_rlp = RasterLoadParams


def _full_tyx_bins(
    tiles: GeoboxTiles, nsrcs=1, nt=1
) -> Dict[tuple[int, int, int], list[int]]:
    return {idx: list(range(nsrcs)) for idx in np.ndindex((nt, *tiles.shape.yx))}  # type: ignore


def _num_chunks(chunk: int, sz: int) -> int:
    return (sz + chunk - 1) // chunk


# bands,extra_coords,extra_dims,expect
rlp_fixtures = [
    [
        # Y,X only
        {"a": _rlp("uint8")},
        None,
        None,
        {"a": _sn(dims=dims, shape=shape)},
    ],
    [
        # Y,X,B coords only, no dims
        {"a": _rlp("uint8", dims=("y", "x", "B"))},
        [FixedCoord("B", ["r", "g", "b"])],
        None,
        {"a": _sn(dims=(*dims, "B"), shape=(*shape, 3))},
    ],
    [
        # Y,X,B dims only
        {"a": _rlp("uint8", dims=("y", "x", "W"))},
        None,
        {"W": 4},
        {"a": _sn(dims=(*dims, "W"), shape=(*shape, 4))},
    ],
    [
        # B,Y,X dims only
        {"a": _rlp("uint8", dims=("W", "y", "x"))},
        None,
        {"W": 4},
        {"a": _sn(dims=(dims[0], "W", *dims[1:]), shape=(shape[0], 4, *shape[1:]))},
    ],
    [
        # Y,X,B coords and dims
        {"a": _rlp("uint16", dims=("y", "x", "W"))},
        [FixedCoord("W", ["r", "g", "b", "a"])],
        {"W": 4},
        {"a": _sn(dims=(*dims, "W"), shape=(*shape, 4))},
    ],
]


def check_xx(
    xx,
    bands: Dict[str, RasterLoadParams],
    extra_coords: Sequence[FixedCoord] | None,
    extra_dims: Mapping[str, int] | None,
    expect: Mapping[str, _sn],
):
    assert isinstance(xx, xr.Dataset)
    for name, dv in xx.data_vars.items():
        assert isinstance(dv.data, (np.ndarray, da.Array))
        assert name in bands
        assert dv.dtype == bands[name].dtype

    assert set(xx.data_vars) == set(bands)

    for n, e in expect.items():
        assert n in xx.data_vars
        v = xx[n]
        assert v.dims == e.dims
        assert v.shape == e.shape

    if extra_coords is not None:
        for c in extra_coords:
            assert c.name in xx.coords
            assert xx.coords[c.name].shape == (len(c.values),)

    if extra_dims is not None:
        for n, s in extra_dims.items():
            assert n in xx.dims
            assert n in xx.sizes
            assert s == xx.sizes[n]


@pytest.mark.parametrize("bands,extra_coords,extra_dims,expect", rlp_fixtures)
def test_mk_dataset(
    bands: Dict[str, RasterLoadParams],
    extra_coords: Sequence[FixedCoord] | None,
    extra_dims: Mapping[str, int] | None,
    expect: Mapping[str, _sn],
):
    assert gbox.crs == "EPSG:4326"
    template = RasterGroupMetadata(
        {
            (k, 1): RasterBandMetadata(b.dtype, b.fill_value, dims=b.dims)
            for k, b in bands.items()
        },
        extra_dims={} if extra_dims is None else {**extra_dims},
        extra_coords=extra_coords or (),
    )
    xx = mk_dataset(
        gbox,
        tss,
        bands=bands,
        template=template,
    )
    check_xx(xx, bands, extra_coords, extra_dims, expect)


@pytest.mark.parametrize("bands,extra_coords,extra_dims,expect", rlp_fixtures)
@pytest.mark.parametrize("chunk_extra_dims", [False, True])
@pytest.mark.parametrize("mode", ["auto", "concurrency"])
def test_dask_builder(
    bands: Dict[str, RasterLoadParams],
    extra_coords: Sequence[FixedCoord] | None,
    extra_dims: Mapping[str, int] | None,
    expect: Mapping[str, _sn],
    chunk_extra_dims: bool,
    mode,
):
    _bands = {
        k: RasterBandMetadata(b.dtype, b.fill_value, dims=b.dims)
        for k, b in bands.items()
    }
    extra_dims = {**extra_dims} if extra_dims is not None else {}
    rgm = RasterGroupMetadata(
        {(k, 1): b for k, b in _bands.items()},
        extra_dims=extra_dims,
        extra_coords=extra_coords or [],
    )

    rdr = FakeReaderDriver(rgm, parser=FakeMDPlugin(rgm, None))
    rdr_env = rdr.capture_env()

    template = RasterGroupMetadata(
        {(k, 1): b for k, b in _bands.items()},
        aliases={},
        extra_dims=extra_dims,
        extra_coords=extra_coords or (),
    )
    src_mapper = {
        k: RasterSource("file:///tmp/a.tif", meta=b) for k, b in _bands.items()
    }
    srcs = [src_mapper, src_mapper, src_mapper]
    tyx_bins = _full_tyx_bins(gbt, nsrcs=len(srcs), nt=len(tss))

    chunks = {"time": 1}
    if chunk_extra_dims:
        chunks = {k: 1 for k in extra_dims}

    builder = DaskGraphBuilder(
        bands,
        template=template,
        srcs=srcs,
        tyx_bins=tyx_bins,
        gbt=gbt,
        env=rdr_env,
        rdr=rdr,
        chunks=chunks,
        mode=mode,
    )

    xx = builder.build(gbox, tss, bands)
    check_xx(xx, bands, extra_coords, extra_dims, expect)

    (yy,) = dask.compute(xx, scheduler="synchronous")
    check_xx(yy, bands, extra_coords, extra_dims, expect)

    xx_direct = chunked_load(bands, template, srcs, tyx_bins, gbt, tss, rdr_env, rdr)
    check_xx(xx_direct, bands, extra_coords, extra_dims, expect)

    xx_dasked = chunked_load(
        bands, template, srcs, tyx_bins, gbt, tss, rdr_env, rdr, chunks={}
    )
    check_xx(xx_dasked, bands, extra_coords, extra_dims, expect)


@pytest.mark.parametrize(
    "cfg,fallback,expect",
    [
        ({}, "uint8", "uint8"),
        ({}, "float64", "float64"),
        (None, "float64", "float64"),
        ({"a": _rlp("uint16")}, "float32", "uint16"),
        ({"a": _rlp("uint16"), "b": _rlp("int32")}, "float32", "int32"),
        ({"a": _rlp()}, "float32", "float32"),
    ],
)
def test_largest_dtype(cfg, fallback, expect):
    assert _largest_dtype(cfg, fallback) == expect


@pytest.mark.parametrize(
    "base_shape,chunks,extra_dims,expect",
    [
        ((1, 200, 300), {}, None, ((1,), (200,), (300,))),
        ((1, 200, 300), {"y": 100}, None, ((1,), (100, 100), (300,))),
        (
            (3, 200, 300),
            {"y": 100, "x": 200, "time": 2},
            None,
            ((2, 1), (100, 100), (200, 100)),
        ),
        ((1, 200, 300), {}, {"b": 3}, ((1,), (200,), (300,), (3,))),
        ((1, 200, 300), {"b": 1}, {"b": 3}, ((1,), (200,), (300,), (1, 1, 1))),
    ],
)
@pytest.mark.parametrize("dtype", [None, "uint8", "uint16"])
def test_resolve_chunks(
    base_shape: tuple[int, int, int],
    chunks: Mapping[str, int | Literal["auto"]],
    extra_dims: Mapping[str, int] | None,
    expect: tuple[int, ...],
    dtype: Any | None,
):
    normed_chunks = resolve_chunks(base_shape, chunks, dtype, extra_dims)
    assert isinstance(normed_chunks, tuple)
    assert len(normed_chunks) == len(base_shape) + len(extra_dims or {})
    assert all(
        (isinstance(ii, tuple) and isinstance(ii[0], int)) for ii in normed_chunks
    )

    if expect is not None:
        assert normed_chunks == expect


def test_resolve_chunk_shape():
    # pylint: disable=redefined-outer-name
    nt = 7
    gbox = GeoBox.from_bbox((-180, -90, 180, 90), shape=(33, 77), tight=True)
    yx_shape = gbox.shape.yx
    assert resolve_chunk_shape(nt, gbox, {}) == (1, *yx_shape)
    assert resolve_chunk_shape(nt, gbox, {"time": 3}) == (3, *yx_shape)
    assert resolve_chunk_shape(nt, gbox, {"y": 10, "x": 20}) == (1, 10, 20)
    assert resolve_chunk_shape(
        nt,
        gbox,
        dict(zip(gbox.dimensions, [10, 20])),
    ) == (1, 10, 20)

    # extra chunks without extra_dims should be ignored
    assert resolve_chunk_shape(nt, gbox, {"y": 10, "x": 20, "b": 3}) == (1, 10, 20)

    # extra_dims and chunking
    assert resolve_chunk_shape(
        nt, gbox, {"y": 10, "x": 20, "b": 3}, extra_dims={"b": 100}
    ) == (1, 10, 20, 3)

    # extra_dims but no chunking
    assert resolve_chunk_shape(
        nt,
        gbox,
        {"y": 10, "x": 20},
        extra_dims={"b": 100},
    ) == (1, 10, 20, 100)


@pytest.mark.parametrize(
    "chunks,dims,nt,nsrcs,extra_dims",
    [
        ({}, (), 3, 1, {}),
        ({"time": 2}, (), 3, 1, {}),
        ({"time": 2, "y": 80, "x": 80}, (), 3, 1, {}),
        ({"b": 2, "y": 80, "x": 80}, ("b", "y", "x"), 2, 1, {"b": 5}),
        ({"time": 2, "y": 100, "b": 1}, ("y", "x", "b"), 4, 3, {"b": 4}),
        ({"time": 2, "y": 100, "b": 1}, ("y", "x", "b"), 1, 3, {"b": 4}),
    ],
)
def test_load_tasks(
    chunks: Mapping[str, int],
    dims: tuple[str, ...],
    extra_dims: Mapping[str, int],
    nt: int,
    nsrcs: int,
):
    var_name = "xx"
    cfg = {var_name: RasterLoadParams("uint8", dims=dims)}

    ydim = 1 + (dims.index("y") if dims else 0)
    assert ydim in (1, 2)

    _nt, ny, nx, *extra_chunks = resolve_chunk_shape(
        nt, gbox, chunks, extra_dims=extra_dims
    )
    assert len(extra_chunks) in (0, 1)
    assert _nt == min(chunks.get("time", 1), nt)
    nt_chunks = _num_chunks(_nt, nt)
    nb_chunks = 1
    if extra_chunks:
        nb_chunks = _num_chunks(extra_chunks[0], list(extra_dims.values())[0])

    gbt = GeoboxTiles(gbox, (ny, nx))

    tyx_bins = _full_tyx_bins(gbt, nsrcs=nsrcs, nt=nt)
    assert len(tyx_bins) == nt * gbt.shape.y * gbt.shape.x

    tasks = load_tasks(cfg, tyx_bins, gbt, nt=nt, chunks=chunks, extra_dims=extra_dims)
    tasks = list(tasks)
    assert len(tasks) == nt_chunks * gbt.shape.y * gbt.shape.x * nb_chunks

    for t in tasks:
        assert t.band == var_name
        assert t.gbt is gbt
        assert t.idx_tyx in tyx_bins
        assert t.ydim == ydim
        assert len(t.srcs) > 0
        assert isinstance(t.srcs[0], list)
        assert len(t.idx) == len(t.postfix_dims) + len(t.prefix_dims) + 2 + 1

        if dims:
            assert len(t.postfix_dims) + len(t.prefix_dims) == len(dims) - 2
            assert len(t.idx) == len(dims) + 1
        else:
            assert t.idx == t.idx_tyx
            assert len(t.postfix_dims) + len(t.prefix_dims) == 0

        assert gbox.enclosing(t.dst_gbox.boundingbox) == t.dst_gbox
        assert gbox[t.dst_roi[ydim : ydim + 2]] == t.dst_gbox
