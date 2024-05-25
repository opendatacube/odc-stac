# pylint: disable=missing-function-docstring,missing-module-docstring,too-many-statements,too-many-locals
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace as _sn
from typing import Dict, Mapping, Sequence

import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox, GeoboxTiles

from . import chunked_load
from ._builder import DaskGraphBuilder, mk_dataset, resolve_chunk_shape
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
def test_dask_builder(
    bands: Dict[str, RasterLoadParams],
    extra_coords: Sequence[FixedCoord] | None,
    extra_dims: Mapping[str, int] | None,
    expect: Mapping[str, _sn],
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

    builder = DaskGraphBuilder(
        bands,
        template=template,
        srcs=srcs,
        tyx_bins=tyx_bins,
        gbt=gbt,
        env=rdr_env,
        rdr=rdr,
        chunks={"time": 1},
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


def test_resolve_chunk_shape():
    # pylint: disable=redefined-outer-name
    nt = 7
    gbox = GeoBox.from_bbox((-180, -90, 180, 90), shape=(33, 77), tight=True)
    yx_shape = gbox.shape.yx
    assert resolve_chunk_shape(nt, gbox, {}) == (1, *yx_shape)
    assert resolve_chunk_shape(nt, gbox, {"time": 3}) == (3, *yx_shape)
    assert resolve_chunk_shape(nt, gbox, {"y": 10, "x": 20}) == (1, 10, 20)

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
