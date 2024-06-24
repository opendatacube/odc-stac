"""
Tests for the in-memory reader driver
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr
from dask import is_dask_collection
from dask.base import tokenize
from odc.geo.data import country_geom
from odc.geo.gcp import GCPGeoBox
from odc.geo.geobox import GeoBox
from odc.geo.xr import ODCExtensionDa, ODCExtensionDs, rasterize

from odc.loader._zarr import (
    Context,
    XrMemReader,
    XrMemReaderDask,
    XrMemReaderDriver,
    raster_group_md,
)
from odc.loader.types import (
    FixedCoord,
    RasterGroupMetadata,
    RasterLoadParams,
    RasterSource,
)

# pylint: disable=missing-function-docstring,use-implicit-booleaness-not-comparison,protected-access
# pylint: disable=too-many-locals,too-many-statements,redefined-outer-name,import-outside-toplevel


@pytest.fixture
def sample_ds() -> xr.Dataset:
    poly = country_geom("AUS", 3857)
    gbox = GeoBox.from_geopolygon(poly, resolution=10_000)
    xx = rasterize(poly, gbox).astype("int16")
    xx.attrs["units"] = "uu"
    xx.attrs["nodata"] = -33

    return xx.to_dataset(name="xx")


def test_mem_reader(sample_ds: xr.Dataset) -> None:
    fake_item = object()

    assert isinstance(sample_ds.odc, ODCExtensionDs)
    gbox = sample_ds.odc.geobox
    assert gbox is not None
    assert isinstance(gbox, GeoBox)

    driver = XrMemReaderDriver(sample_ds)

    assert driver.md_parser is not None

    md = driver.md_parser.extract(fake_item)
    assert isinstance(md, RasterGroupMetadata)
    assert len(md.bands) == 1
    assert ("xx", 1) in md.bands
    assert md.bands[("xx", 1)].data_type == "int16"
    assert md.bands[("xx", 1)].units == "uu"
    assert md.bands[("xx", 1)].nodata == -33
    assert md.bands[("xx", 1)].dims == ()
    assert len(md.aliases) == 0
    assert md.extra_dims == {}
    assert md.extra_coords == []

    ds = sample_ds.copy()
    xx = ds.xx
    yy = xx.astype("uint8", keep_attrs=False).rename("yy")
    yy = yy.expand_dims("band", 2)
    yy = xr.concat([yy, yy + 1, yy + 2], "band").assign_coords(band=["r", "g", "b"])
    yy.band.attrs["units"] = "CC"

    assert yy.odc.geobox == gbox

    ds["yy"] = yy
    ds["zz"] = yy.transpose("band", "y", "x")

    driver = XrMemReaderDriver(ds)
    assert driver.md_parser is not None
    assert driver.dask_reader is not None
    md = driver.md_parser.extract(fake_item)

    assert isinstance(md, RasterGroupMetadata)
    assert len(md.bands) == 3
    assert ("xx", 1) in md.bands
    assert ("yy", 1) in md.bands
    assert ("zz", 1) in md.bands
    assert md.bands[("xx", 1)].data_type == "int16"
    assert md.bands[("xx", 1)].units == "uu"
    assert md.bands[("xx", 1)].nodata == -33
    assert md.bands[("xx", 1)].dims == ()
    assert md.bands[("yy", 1)].data_type == "uint8"
    assert md.bands[("yy", 1)].units == "1"
    assert md.bands[("yy", 1)].nodata is None
    assert md.bands[("yy", 1)].dims == ("y", "x", "band")
    assert md.bands[("zz", 1)].dims == ("band", "y", "x")

    assert len(md.aliases) == 0
    assert md.extra_dims == {"band": 3}
    assert len(md.extra_coords) == 1

    (coord,) = md.extra_coords
    assert coord.name == "band"
    assert coord.units == "CC"
    assert coord.dim == "band"
    assert isinstance(coord.values, np.ndarray)
    assert coord.values.tolist() == ["r", "g", "b"]

    oo: ODCExtensionDa = ds.yy.odc
    assert isinstance(oo.geobox, GeoBox)

    env = driver.capture_env()
    ctx = driver.new_load(oo.geobox)
    assert isinstance(env, dict)
    srcs = {
        n: RasterSource(
            f"mem://{n}",
            meta=md.bands[n, 1],
            driver_data=driver.md_parser.driver_data(fake_item, (n, 1)),
        )
        for n, _ in md.bands
    }
    cfgs = {n: RasterLoadParams.same_as(src) for n, src in srcs.items()}

    with driver.restore_env(env, ctx) as _ctx:
        assert _ctx is not None

        loaders = {n: driver.open(srcs[n], ctx) for n in srcs}
        assert set(loaders) == set(srcs)

        for n, loader in loaders.items():
            assert isinstance(loader, XrMemReader)
            roi, pix = loader.read(cfgs[n], gbox)
            assert roi == (slice(None), slice(None))
            assert isinstance(pix, np.ndarray)
            if n == "xx":
                assert pix.dtype == np.int16
                assert pix.shape == gbox.shape.yx
            elif n == "yy":
                assert pix.dtype == np.uint8
                assert pix.shape == (*gbox.shape.yx, 3)
            elif n == "zz":
                assert pix.shape == (3, *gbox.shape.yx)

        loader = loaders["yy"]
        roi, pix = loader.read(cfgs["yy"], gbox, selection=np.s_[:2])
        assert pix.shape == (*gbox.shape.yx, 2)

        loader = loaders["zz"]
        roi, pix = loader.read(cfgs["zz"], gbox, selection=np.s_[:2])
        assert pix.shape == (2, *gbox.shape.yx)


def test_raster_group_md():
    rgm = raster_group_md(xr.Dataset())
    assert rgm.bands == {}
    assert rgm.aliases == {}
    assert rgm.extra_dims == {}

    coord = FixedCoord("band", ["r", "g", "b"], dim="band")

    rgm = raster_group_md(
        xr.Dataset(), base=RasterGroupMetadata({}, {}, {"band": 3}, [])
    )
    assert rgm.extra_dims == {"band": 3}
    assert len(rgm.extra_coords) == 0

    rgm = raster_group_md(
        xr.Dataset(), base=RasterGroupMetadata({}, extra_coords=[coord])
    )
    assert rgm.extra_dims == {}
    assert rgm.extra_dims_full() == {"band": 3}
    assert len(rgm.extra_coords) == 1
    assert rgm.extra_coords[0] == coord


def test_memreader_zarr(sample_ds: xr.Dataset):
    assert isinstance(sample_ds.odc, ODCExtensionDs)
    assert "xx" in sample_ds

    zarr = pytest.importorskip("zarr")
    assert zarr is not None

    _gbox = sample_ds.odc.geobox
    chunks = None
    assert _gbox is not None
    gbox = _gbox.approx if isinstance(_gbox, GCPGeoBox) else _gbox

    md_store: dict[str, bytes] = {}
    chunk_store: dict[str, bytes] = {}
    sample_ds.to_zarr(md_store, chunk_store, compute=False, consolidated=True)

    assert ".zmetadata" in md_store
    zmd = json.loads(md_store[".zmetadata"])["metadata"]

    src = RasterSource(
        "file:///tmp/no-such-dir/xx.zarr",
        subdataset="xx",
        driver_data=zmd,
    )
    assert src.driver_data is zmd

    cfg = RasterLoadParams.same_as(src)

    ctx = Context(gbox, chunks)
    rdr = XrMemReader(src, ctx)

    roi, xx = rdr.read(cfg, gbox)
    assert isinstance(xx, np.ndarray)
    assert xx.shape == gbox[roi].shape.yx
    assert gbox == gbox[roi]

    tk = tokenize(src, cfg, gbox)
    ctx = Context(gbox, {})
    rdr = XrMemReaderDask().open(src, ctx, layer_name=f"xx-{tk}", idx=0)
    assert isinstance(rdr, XrMemReaderDask)
    assert rdr._src is not None
    assert rdr._src._chunks == {}

    fut = rdr.read(cfg, gbox)
    assert is_dask_collection(fut)

    roi, xx = fut.compute(scheduler="synchronous")
    assert isinstance(xx, np.ndarray)
    assert roi == (slice(None), slice(None))
    assert xx.shape == gbox.shape.yx
