"""
Test for SQS to DC tool
"""
import json
from pathlib import Path
import pytest
from datacube.utils import documents


TEST_DATA_FOLDER: Path = Path(__file__).parent.joinpath("data")
LANDSAT_STAC: str = "ga_ls8c_ard_3-1-0_088080_2020-05-25_final.stac-item.json"
LANDSAT_ODC: str = "ga_ls8c_ard_3-1-0_088080_2020-05-25_final.odc-metadata.yaml"
SENTINEL_STAC: str = "S2A_28QCH_20200714_0_L2A.json"
SENTINEL_STAC_MS: str = "S2B_MSIL2A_20190629T212529_R043_T06VVN_20201006T080531.json"
SENTINEL_ODC: str = "S2A_28QCH_20200714_0_L2A.odc-metadata.json"
USGS_LANDSAT_STAC: str = "LC08_L2SR_081119_20200101_20200823_02_T2.json"
LIDAR_STAC: str = "lidar_dem.json"

@pytest.fixture
def usgs_landsat_stac():
    with TEST_DATA_FOLDER.joinpath(USGS_LANDSAT_STAC).open("r") as f:
        return json.load(f)


@pytest.fixture
def landsat_stac():
    with TEST_DATA_FOLDER.joinpath(LANDSAT_STAC).open("r") as f:
        metadata = json.load(f)
    return metadata


@pytest.fixture
def lidar_stac():
    with TEST_DATA_FOLDER.joinpath(LIDAR_STAC).open("r") as f:
        metadata = json.load(f)
    return metadata


@pytest.fixture
def landsat_odc():
    metadata = yield from documents.load_documents(
        TEST_DATA_FOLDER.joinpath(LANDSAT_ODC)
    )
    return metadata


@pytest.fixture
def sentinel_stac():
    with TEST_DATA_FOLDER.joinpath(SENTINEL_STAC).open("r") as f:
        metadata = json.load(f)
    return metadata


@pytest.fixture
def sentinel_stac_ms():
    with TEST_DATA_FOLDER.joinpath(SENTINEL_STAC_MS).open("r") as f:
        metadata = json.load(f)
    return metadata


@pytest.fixture
def sentinel_odc():
    with TEST_DATA_FOLDER.joinpath(SENTINEL_ODC).open("r") as f:
        metadata = json.load(f)
    return metadata
