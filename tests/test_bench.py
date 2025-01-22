# pylint: disable=wrong-import-order,wrong-import-position,
# pylint: disable=redefined-outer-name,missing-function-docstring,missing-module-docstring
import pytest

distributed = pytest.importorskip("distributed")

from unittest.mock import MagicMock

import xarray
import numpy as np
from distributed import Client
from odc.geo.xr import ODCExtension

from odc.stac.bench import (
    BenchLoadParams,
    collect_context_info,
    load_from_json,
    run_bench,
)

CFG = {
    "*": {
        "warnings": "ignore",
        # for every asset in every product default to uint16 with nodata=0
        "assets": {"*": {"data_type": "uint16", "nodata": 0}},
    }
}


@pytest.fixture
def fake_dask_client(monkeypatch):
    cc = MagicMock()
    cc.scheduler_info.return_value = {
        "type": "Scheduler",
        "id": "Scheduler-80d943db-16f6-4476-a51a-64d57a287e9b",
        "address": "inproc://10.10.10.10/1281505/1",
        "services": {"dashboard": 8787},
        "started": 1638320006.6135786,
        "workers": {
            "inproc://10.10.10.10/1281505/4": {
                "type": "Worker",
                "id": 0,
                "host": "10.1.1.140",
                "resources": {},
                "local_directory": "/tmp/dask-worker-space/worker-uhq1b9bh",
                "name": 0,
                "nthreads": 2,
                "memory_limit": 524288000,
                "last_seen": 1638320007.2504623,
                "services": {"dashboard": 38439},
                "metrics": {
                    "executing": 0,
                    "in_memory": 0,
                    "ready": 0,
                    "in_flight": 0,
                    "bandwidth": {"total": 100000000, "workers": {}, "types": {}},
                    "spilled_nbytes": 0,
                    "cpu": 0.0,
                    "memory": 145129472,
                    "time": 1638320007.2390554,
                    "read_bytes": 0.0,
                    "write_bytes": 0.0,
                    "read_bytes_disk": 0.0,
                    "write_bytes_disk": 0.0,
                    "num_fds": 82,
                },
                "nanny": None,
            }
        },
    }
    cc.cancel.return_value = None
    cc.restart.return_value = cc
    cc.persist = lambda x: x
    cc.compute = lambda x: x

    monkeypatch.setattr(distributed, "wait", MagicMock())
    yield cc


@pytest.fixture(scope="module")
def dask_client():
    client = Client(
        n_workers=1,
        threads_per_worker=2,
        memory_limit="500MiB",
        local_directory="/tmp/",
        memory_target_fraction=False,
        memory_spill_fraction=False,
        memory_pause_fraction=False,
        dashboard_address=None,
        processes=False,
    )
    yield client
    client.shutdown()
    del client


@pytest.mark.skipif(
    not pytest.importorskip("stackstac"), reason="stackstac not installed"
)
def test_load_from_json_stackstac(fake_dask_client, bench_site1, bench_site2):
    dask_client = fake_dask_client
    params = BenchLoadParams(
        scenario="test1",
        method="stackstac",
        bands=("B04", "B02", "B03"),
        chunks=(2048, 2048),
        resampling="nearest",
        extra={
            "odc-stac": {"groupby": "solar_day", "stac_cfg": CFG},
            "stackstac": {
                "dtype": "uint16",
                "fill_value": np.uint16(0),
                "rescale": False,
            },
        },
    )
    xx = load_from_json(bench_site1, params)
    assert "band" in xx.dims
    assert xx.shape == (1, 3, 90978, 10980)
    assert xx.dtype == "uint16"
    assert xx.spec.epsg == 32735

    yy = load_from_json(
        bench_site1, params.with_method("odc-stac"), geobox=xx.odc.geobox
    )

    rrx = collect_context_info(dask_client, xx)
    rry = collect_context_info(dask_client, yy)
    assert rrx.shape == rry.shape
    assert rrx == rry

    xx = load_from_json(bench_site2, params)
    assert "band" in xx.dims
    assert xx.dtype == "uint16"
    assert xx.spec.epsg == 32735

    params.crs = "epsg:32736"
    xx = load_from_json(bench_site2, params)
    assert "band" in xx.dims
    assert xx.dtype == "uint16"
    assert xx.spec.epsg == 32736

    with pytest.raises(ValueError):
        load_from_json(bench_site1, params.with_method("wroNg"))


def test_bench_context(fake_dask_client, bench_site1, bench_site2):
    params = BenchLoadParams(
        scenario="test1",
        method="odc-stac",
        bands=("red", "green", "blue"),
        chunks=(2048, 2048),
        extra={"odc-stac": {"groupby": "solar_day", "stac_cfg": CFG}},
    )
    xx = load_from_json(bench_site1, params)
    nt, ny, nx = xx.red.shape
    nb = len(xx.data_vars)

    # Check normal case Dataset, with time coords
    rr = collect_context_info(
        fake_dask_client, xx, method=params.method, scenario="site1"
    )
    assert isinstance(xx.odc, ODCExtension)
    assert rr.shape == (nt, nb, ny, nx)
    assert rr.chunks == (1, 1, 2048, 2048)
    assert rr.crs == f"epsg:{xx.odc.geobox.crs.epsg}"
    assert rr.crs == xx.odc.geobox.crs
    assert rr.nthreads == 2
    assert rr.total_ram == 500 * (1 << 20)

    header_txt = rr.render_txt()
    assert "T.slice   : 2020-06-06" in header_txt
    assert f"Data      : 1.3.{ny}.{nx}.uint16,  5.58 GiB" in header_txt

    run_txt = rr.render_timing_info((0, 0.1, 30))
    assert isinstance(run_txt, str)

    pd_dict = rr.to_pandas_dict()
    assert pd_dict["resolution"] == rr.resolution
    assert pd_dict["data"] == f"1.3.{ny}.{nx}.uint16"
    assert pd_dict["chunks_x"] == 2048
    assert pd_dict["chunks_y"] == 2048

    # Check DataArray case
    rr = collect_context_info(
        fake_dask_client, xx.red, method="odc-stac", scenario="site1"
    )
    assert rr.shape == (nt, 1, ny, nx)
    assert rr.crs == xx.odc.geobox.crs

    # Check Dataset with 0 dimension time axis and extras field
    rr = collect_context_info(
        fake_dask_client,
        xx.isel(time=0),
        method=params.method,
        scenario=params.scenario,
        extras={"custom": 2},
    )
    assert rr.extras == {"custom": 2}
    assert rr.shape == (1, nb, ny, nx)

    header_txt = rr.render_txt()
    assert "GEO       : epsg:32735" in header_txt
    assert "T.slice   : 2020-06-06" in header_txt

    # Check no time info at all
    rr = collect_context_info(
        fake_dask_client,
        xx.isel(time=0, drop=True),
        method=params.method,
        scenario=params.scenario,
    )
    assert rr.shape == (nt, nb, ny, nx)
    assert rr.dtype == xx.red.dtype
    assert rr.temporal_id == "-"

    # Check wrong type
    with pytest.raises(ValueError):
        collect_context_info(fake_dask_client, "wrong input type")  # type: ignore

    # Check multi-time axis
    xx = load_from_json(bench_site2, params)
    nt, ny, nx = xx.red.shape
    nb = len(xx.data_vars)

    assert nt > 1

    rr = collect_context_info(
        fake_dask_client,
        xx,
        method=params.method,
        scenario=params.scenario,
    )
    assert rr.shape == (nt, nb, ny, nx)
    assert rr.temporal_id == "2020-06-01__2020-07-31"

    # Check missing GEO info
    no_geo = _strip_geo(xx.red)
    assert no_geo.odc.geobox is None or no_geo.odc.geobox.crs is None
    with pytest.raises(ValueError):
        # no geobox
        collect_context_info(fake_dask_client, no_geo)


def _strip_geo(xx: xarray.DataArray) -> xarray.DataArray:
    no_geo = xx.drop_vars("spatial_ref")
    no_geo.attrs.pop("crs", None)
    no_geo.attrs.pop("grid_mapping", None)
    no_geo.encoding.pop("grid_mapping", None)
    no_geo.x.attrs.pop("crs", None)
    no_geo.y.attrs.pop("crs", None)
    # get rid of cached geobox
    no_geo = xarray.DataArray(
        no_geo.data,
        coords=no_geo.coords,
        dims=no_geo.dims,
        attrs=no_geo.attrs,
    )
    assert no_geo.odc.geobox is None or no_geo.odc.geobox.crs is None
    return no_geo


def test_run_bench(fake_dask_client, bench_site1, capsys):
    dask_client = fake_dask_client
    params = BenchLoadParams(
        scenario="test1",
        method="odc-stac",
        bands=("red", "green", "blue"),
        chunks=(2048, 2048),
        extra={"odc-stac": {"groupby": "solar_day", "stac_cfg": CFG}},
    )
    xx = load_from_json(bench_site1, params)

    rr, timing = run_bench(xx, dask_client, 10)

    assert rr.scenario == params.scenario
    assert rr.method == params.method
    assert len(timing) == 10
    _io = capsys.readouterr()
    assert len(_io.out) > 0


def test_bench_params_json():
    params = BenchLoadParams(
        scenario="test1",
        method="odc-stac",
        bands=("red", "green", "blue"),
        chunks=(100, 200),
        extra={"odc-stac": {"groupby": "solar_day", "stac_cfg": CFG}},
    )

    assert params == BenchLoadParams.from_json(params.to_json())
    assert params.to_json() == BenchLoadParams.from_json(params.to_json()).to_json()

    # function should round-trip too
    params.patch_url = load_from_json
    assert params == BenchLoadParams.from_json(params.to_json())
