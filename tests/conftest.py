"""
Test for SQS to DC tool
"""

import json
from pathlib import Path

import pystac
import pystac.collection
import pystac.item
import pytest
from odc.geo.data import country_geom

TEST_DATA_FOLDER: Path = Path(__file__).parent.joinpath("data")
PARTIAL_PROJ_STAC: str = "only_crs_proj.json"
GA_LANDSAT_STAC: str = "ga_ls8c_ard_3-1-0_088080_2020-05-25_final.stac-item.json"
SENTINEL_STAC_COLLECTION: str = "sentinel-2-l2a.collection.json"
SENTINEL_STAC: str = "S2A_28QCH_20200714_0_L2A.json"
SENTINEL_STAC_MS: str = "S2B_MSIL2A_20190629T212529_R043_T06VVN_20201006T080531.json"
SENTINEL_STAC_MS_RASTER_EXT: str = (
    "S2B_MSIL2A_20190629T212529_R043_T06VVN_20201006T080531_raster_ext.json"
)
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
def sentinel_stac():
    return pystac.item.Item.from_file(str(TEST_DATA_FOLDER.joinpath(SENTINEL_STAC)))


@pytest.fixture
def sentinel_stac_ms_json():
    with TEST_DATA_FOLDER.joinpath(SENTINEL_STAC_MS).open("r", encoding="utf") as f:
        return json.load(f)


@pytest.fixture
def bench_site1():
    with TEST_DATA_FOLDER.joinpath(BENCH_SITE1).open("r", encoding="utf") as f:
        return _strip_links(json.load(f))


@pytest.fixture
def bench_site2():
    with TEST_DATA_FOLDER.joinpath(BENCH_SITE2).open("r", encoding="utf") as f:
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
    assert isinstance(item, pystac.Item)
    item.remove_links("self")
    return item


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


def _strip_links(gjson):
    for item in gjson["features"]:
        item["links"] = []
    return gjson


@pytest.fixture()
def gpd_iso3():
    def _get(iso3: str, crs=None):
        return country_geom(iso3.upper(), crs=crs)

    yield _get
