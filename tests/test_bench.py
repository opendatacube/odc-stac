from typing import Literal
import pytest
from odc.stac.bench import BenchmarkContext, collect_context_info, load_from_json
from distributed import Client, client

CFG = {"*": {"warnings": "ignore"}}


@pytest.fixture(scope="module")
def dask_client():
    yield Client(
        n_workers=1,
        threads_per_worker=2,
        memory_limit="500MiB",
        local_directory="/tmp/",
        memory_target_fraction=False,
        memory_spill_fraction=False,
        memory_pause_fraction=False,
    )


def test_load_from_json_stackstac(dask_client, bench_site1, bench_site2):
    xx = load_from_json(
        bench_site1,
        method="stackstac",
        assets=["B04", "B02", "B03"],
        dtype="uint16",
        chunksize=2048,
    )
    assert "band" in xx.dims
    assert xx.shape == (1, 3, 90978, 10980)
    assert xx.dtype == "uint16"
    assert xx.spec.epsg == 32735

    yy = load_from_json(
        bench_site1,
        method="odc-stac",
        bands=["B04", "B02", "B03"],
        chunks={"x": 2048, "y": 2048},
        groupby="solar_day",
        stac_cfg=CFG,
    )

    rrx = collect_context_info(dask_client, xx)
    rry = collect_context_info(dask_client, yy)
    assert rrx.shape == rry.shape
    assert rrx == rry

    xx = load_from_json(
        bench_site2,
        method="stackstac",
        assets=["B04", "B02", "B03"],
        dtype="uint16",
    )
    assert "band" in xx.dims
    assert xx.dtype == "uint16"
    assert xx.spec.epsg == 32735

    with pytest.raises(ValueError):
        load_from_json(bench_site1, "wroNg")


def test_bench_context(dask_client, bench_site1, bench_site2):
    xx = load_from_json(
        bench_site1,
        method="odc-stac",
        bands=["red", "green", "blue"],
        chunks={"x": 2048, "y": 2048},
        groupby="solar_day",
        stac_cfg=CFG,
    )
    nt, ny, nx = xx.red.shape
    nb = len(xx.data_vars)

    # Check normal case Dataset, with time coords
    rr = collect_context_info(dask_client, xx, method="odc-stac", scenario="site1")
    assert rr.shape == (nt, nb, ny, nx)
    assert rr.chunks == (1, 1, 2048, 2048)
    assert rr.crs == f"epsg:{xx.geobox.crs.epsg}"
    assert rr.crs == xx.geobox.crs
    assert rr.nthreads == 2
    assert rr.total_ram == 500 * (1 << 20)

    header_txt = rr.render_txt()
    print(header_txt)
    assert "T.slice   : 2020-06-06" in header_txt
    assert "Data      : 1.3.90978.10980.uint16,  5.58 GiB" in header_txt

    run_txt = rr.render_timing_info((0, 0.1, 30))
    print(run_txt)

    # Check DataArray case
    rr = collect_context_info(dask_client, xx.red, method="odc-stac", scenario="site1")
    assert rr.shape == (nt, 1, ny, nx)
    assert rr.crs == xx.geobox.crs

    # Check Dataset with 0 dimension time axis and extras field
    rr = collect_context_info(
        dask_client,
        xx.isel(time=0),
        method="odc-stac",
        scenario="site1",
        extras={"custom": 2},
    )
    assert rr.extras == {"custom": 2}
    assert rr.shape == (1, nb, ny, nx)

    header_txt = rr.render_txt()
    assert "GEO       : epsg:32735" in header_txt
    assert "T.slice   : 2020-06-06" in header_txt

    # Check no time info at all
    rr = collect_context_info(
        dask_client,
        xx.isel(time=0, drop=True),
        method="odc-stac",
        scenario="site1",
    )
    assert rr.shape == (nt, nb, ny, nx)
    assert rr.dtype == xx.red.dtype
    assert rr.temporal_id == "-"

    # Check wrong type
    with pytest.raises(ValueError):
        collect_context_info(dask_client, "wrong input type")  # type: ignore

    # Check multi-time axis
    xx = load_from_json(
        bench_site2,
        method="odc-stac",
        bands=["red", "green", "blue"],
        chunks={"x": 2048, "y": 2048},
        groupby="solar_day",
        stac_cfg=CFG,
    )
    nt, ny, nx = xx.red.shape
    nb = len(xx.data_vars)

    assert nt > 1

    rr = collect_context_info(
        dask_client,
        xx,
        method="odc-stac",
        scenario="site1",
    )
    assert rr.shape == (nt, nb, ny, nx)
    assert rr.temporal_id == "2020-06-01__2020-07-31"

    # Check missing GEO info
    no_geo = xx.red.drop_vars("spatial_ref")
    no_geo.attrs.pop("crs", None)
    no_geo.attrs.pop("grid_mapping", None)
    no_geo.x.attrs.pop("crs", None)
    no_geo.y.attrs.pop("crs", None)
    assert no_geo.geobox is None
    with pytest.raises(ValueError):
        # no geobox
        collect_context_info(dask_client, no_geo)
