"""
Test for SQS to DC tool
"""
import json
from functools import partial
from pprint import pformat

import pytest

from pathlib import Path

from datacube.utils import documents
from deepdiff import DeepDiff

from odc.index.stac import stac_transform


TEST_DATA_FOLDER: Path = Path(__file__).parent.joinpath("data")
LANDSAT_STAC: str = "ga_ls8c_ard_3-1-0_088080_2020-05-25_final.stac-item.json"
LANDSAT_ODC: str = "ga_ls8c_ard_3-1-0_088080_2020-05-25_final.odc-metadata.yaml"
SENTINEL_STAC: str = "S2A_28QCH_20200714_0_L2A.json"
SENTINEL_ODC: str = "S2A_28QCH_20200714_0_L2A.odc-metadata.json"

deep_diff = partial(DeepDiff, significant_digits=6, ignore_type_in_groups=[(tuple, list)])


def test_landsat_stac_transform(landsat_stac, landsat_odc):
    actual_doc = stac_transform(landsat_stac)
    do_diff(actual_doc, landsat_odc)


def test_sentinel_stac_transform(sentinel_stac, sentinel_odc):
    actual_doc = stac_transform(sentinel_stac)
    do_diff(actual_doc, sentinel_odc)


def do_diff(actual_doc, expected_doc):

    assert expected_doc['id'] == actual_doc['id']
    assert expected_doc['crs'] == actual_doc['crs']
    assert expected_doc['product']['name'] == actual_doc['product']['name']
    assert expected_doc['label'] == actual_doc['label']

    # Test geometry field
    doc_diff = deep_diff(expected_doc['geometry'], actual_doc['geometry'])
    assert doc_diff == {}, pformat(doc_diff)

    # Test grids field
    doc_diff = deep_diff(expected_doc['grids'], actual_doc['grids'])
    assert doc_diff == {}, pformat(doc_diff)

    # Test measurements field
    doc_diff = deep_diff(expected_doc['measurements'], actual_doc['measurements'])
    assert doc_diff == {}, pformat(doc_diff)

    # Test properties field
    doc_diff = deep_diff(expected_doc['properties'], actual_doc['properties'],
                         exclude_paths=["root['odc:product']",
                                        "root['proj:epsg']",
                                        "root['proj:shape']",
                                        "root['proj:transform']"])
    assert doc_diff == {}, pformat(doc_diff)

    # Test lineage field
    doc_diff = deep_diff(expected_doc['lineage'], actual_doc['lineage'])
    assert doc_diff == {}, pformat(doc_diff)


@pytest.fixture
def landsat_stac():
    with TEST_DATA_FOLDER.joinpath(LANDSAT_STAC).open("r") as f:
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
def sentinel_odc():
    with TEST_DATA_FOLDER.joinpath(SENTINEL_ODC).open("r") as f:
        metadata = json.load(f)
    return metadata
