# pylint: disable=missing-function-docstring,missing-module-docstring,too-many-statements
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

from ._builder import DaskGraphBuilder, mk_dataset
from .testing.fixtures import FakeMDPlugin, FakeReaderDriver
from .types import (
    FixedCoord,
    RasterBandMetadata,
    RasterGroupMetadata,
    RasterLoadParams,
    RasterSource,
)

time = [datetime(2020, 1, 1)]
gbox = GeoBox.from_bbox((-180, -90, 180, 90), shape=(160, 320), tight=True)
gbt = GeoboxTiles(gbox, (80, 80))
shape = (len(time), *gbox.shape.yx)
dims = ("time", *gbox.dimensions)
tyx_bins = {(0, *idx): [0] for idx in np.ndindex(gbt.shape.yx)}
_rlp = RasterLoadParams

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
    xx = mk_dataset(
        gbox,
        time,
        bands=bands,
        extra_coords=extra_coords,
        extra_dims=extra_dims,
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

    builder = DaskGraphBuilder(
        bands,
        template=template,
        srcs=srcs,
        tyx_bins=tyx_bins,
        gbt=gbt,
        env=rdr_env,
        rdr=rdr,
        time_chunks=1,
    )

    xx = builder.build(gbox, time, bands)
    check_xx(xx, bands, extra_coords, extra_dims, expect)

    (yy,) = dask.compute(xx, scheduler="synchronous")
    check_xx(yy, bands, extra_coords, extra_dims, expect)
