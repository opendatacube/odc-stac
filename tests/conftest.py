"""
Test for SQS to DC tool
"""
import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import distributed
import geopandas as gpd
import pystac
import pystac.collection
import pystac.item
import pytest
from datacube.utils import documents

from odc.stac.eo3 import stac2ds

TEST_DATA_FOLDER: Path = Path(__file__).parent.joinpath("data")
PARTIAL_PROJ_STAC: str = "only_crs_proj.json"
GA_LANDSAT_STAC: str = "ga_ls8c_ard_3-1-0_088080_2020-05-25_final.stac-item.json"
GA_LANDSAT_ODC: str = "ga_ls8c_ard_3-1-0_088080_2020-05-25_final.odc-metadata.yaml"
SENTINEL_STAC_COLLECTION: str = "sentinel-2-l2a.collection.json"
SENTINEL_STAC: str = "S2A_28QCH_20200714_0_L2A.json"
SENTINEL_STAC_MS: str = "S2B_MSIL2A_20190629T212529_R043_T06VVN_20201006T080531.json"
SENTINEL_STAC_MS_RASTER_EXT: str = (
    "S2B_MSIL2A_20190629T212529_R043_T06VVN_20201006T080531_raster_ext.json"
)
SENTINEL_ODC: str = "S2A_28QCH_20200714_0_L2A.odc-metadata.json"
USGS_LANDSAT_STAC_v1b: str = "LC08_L2SR_081119_20200101_20200823_02_T2.json"
USGS_LANDSAT_STAC_v1: str = "LC08_L2SP_028030_20200114_20200824_02_T1_SR.json"
USGS_LANDSAT_STAC_v1_1_1: str = "LE07_L2SP_044033_20210329_20210424_02_T1_SR.json"
LIDAR_STAC: str = "lidar_dem.json"
BENCH_SITE1: str = "site1-20200606-tall-strip-africa.geojson"
BENCH_SITE2: str = "site2-2020_jun_jul-35MNM.geojson"

# pylint: disable=redefined-outer-name


@pytest.fixture(scope="session")
def test_data_dir():
    return TEST_DATA_FOLDER


@pytest.fixture
def partial_proj_stac():
    return pystac.item.Item.from_file(str(TEST_DATA_FOLDER.joinpath(PARTIAL_PROJ_STAC)))


@pytest.fixture
def no_bands_stac(partial_proj_stac):
    partial_proj_stac.assets.clear()
    return partial_proj_stac


@pytest.fixture
def usgs_landsat_stac_v1():
    return pystac.item.Item.from_file(
        str(TEST_DATA_FOLDER.joinpath(USGS_LANDSAT_STAC_v1))
    )


@pytest.fixture
def usgs_landsat_stac_v1b():
    return pystac.item.Item.from_file(
        str(TEST_DATA_FOLDER.joinpath(USGS_LANDSAT_STAC_v1b))
    )


@pytest.fixture
def usgs_landsat_stac_v1_1_1():
    return pystac.item.Item.from_file(
        str(TEST_DATA_FOLDER.joinpath(USGS_LANDSAT_STAC_v1_1_1))
    )


@pytest.fixture
def ga_landsat_stac():
    return pystac.item.Item.from_file(str(TEST_DATA_FOLDER.joinpath(GA_LANDSAT_STAC)))


@pytest.fixture
def lidar_stac():
    return pystac.item.Item.from_file(str(TEST_DATA_FOLDER.joinpath(LIDAR_STAC)))


@pytest.fixture
def ga_landsat_odc():
    metadata = yield from documents.load_documents(
        TEST_DATA_FOLDER.joinpath(GA_LANDSAT_ODC)
    )
    return metadata


@pytest.fixture
def sentinel_stac():
    return pystac.item.Item.from_file(str(TEST_DATA_FOLDER.joinpath(SENTINEL_STAC)))


@pytest.fixture
def sentinel_stac_ms_json():
    with TEST_DATA_FOLDER.joinpath(SENTINEL_STAC_MS).open("r") as f:
        return json.load(f)


@pytest.fixture
def bench_site1():
    with TEST_DATA_FOLDER.joinpath(BENCH_SITE1).open("r") as f:
        return _strip_links(json.load(f))


@pytest.fixture
def bench_site2():
    with TEST_DATA_FOLDER.joinpath(BENCH_SITE2).open("r") as f:
        return _strip_links(json.load(f))


@pytest.fixture
def sentinel_stac_ms():
    return pystac.item.Item.from_file(str(TEST_DATA_FOLDER.joinpath(SENTINEL_STAC_MS)))


@pytest.fixture
def sentinel_stac_ms_no_ext(sentinel_stac_ms_json):
    metadata = dict(sentinel_stac_ms_json)
    metadata["stac_extensions"] = []
    return pystac.item.Item.from_dict(metadata)


@pytest.fixture
def sentinel_stac_ms_with_raster_ext():
    return pystac.item.Item.from_file(
        str(TEST_DATA_FOLDER.joinpath(SENTINEL_STAC_MS_RASTER_EXT))
    )


@pytest.fixture
def sentinel_stac_collection():
    return pystac.collection.Collection.from_file(
        str(TEST_DATA_FOLDER.joinpath(SENTINEL_STAC_COLLECTION))
    )


@pytest.fixture
def sentinel_odc():
    return pystac.item.Item.from_file(str(TEST_DATA_FOLDER.joinpath(SENTINEL_ODC)))


@pytest.fixture
def relative_href_only(ga_landsat_stac: pystac.item.Item):
    item = pystac.Item.from_dict(ga_landsat_stac.to_dict())
    item = item.make_asset_hrefs_relative()
    item.remove_links("self")
    return item


@pytest.fixture
def s2_dataset(sentinel_stac_ms_with_raster_ext):
    (ds,) = stac2ds(
        [sentinel_stac_ms_with_raster_ext], cfg={"*": {"warnings": "ignore"}}
    )
    yield ds


@pytest.fixture
def sample_geojson():
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "Kangaroo Island"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [136.351318359375, -35.78217070326606],
                            [136.7303466796875, -36.16448788632062],
                            [137.5323486328125, -36.16005298551352],
                            [137.8179931640625, -35.933540642493114],
                            [138.0816650390625, -36.05798104702501],
                            [138.2025146484375, -35.74205383068035],
                            [137.5653076171875, -35.46066995149529],
                            [136.351318359375, -35.78217070326606],
                        ]
                    ],
                },
            }
        ],
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


def _strip_links(gjson):
    for item in gjson["features"]:
        item["links"] = []
    return gjson


@pytest.fixture()
def gpd_natural_earth():
    yield gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))


@pytest.fixture()
def gpd_iso3(gpd_natural_earth):
    def _get(iso3, crs=None):
        gg = gpd_natural_earth[gpd_natural_earth.iso_a3 == iso3]
        if crs is not None:
            gg = gg.to_crs(crs)
        return gg

    yield _get


@pytest.fixture()
def without_aws_env(monkeypatch):
    for e in os.environ:
        if e.startswith("AWS_"):
            monkeypatch.delenv(e, raising=False)
    yield
