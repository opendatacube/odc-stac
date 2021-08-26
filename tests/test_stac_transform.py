"""
Test for stac_transform
"""
from functools import partial
from pprint import pformat
import pytest
from deepdiff import DeepDiff
from odc.index.stac import stac_transform


deep_diff = partial(
    DeepDiff, significant_digits=6, ignore_type_in_groups=[(tuple, list)]
)


def test_landsat_stac_transform(landsat_stac, landsat_odc):
    actual_doc = stac_transform(landsat_stac)
    do_diff(actual_doc, landsat_odc)


def test_sentinel_stac_transform(sentinel_stac, sentinel_odc):
    actual_doc = stac_transform(sentinel_stac)
    do_diff(actual_doc, sentinel_odc)


def test_usgs_landsat_stac_transform(usgs_landsat_stac):
    _ = stac_transform(usgs_landsat_stac)


def test_lidar_stac_transform(lidar_stac):
    _ = stac_transform(lidar_stac)


def do_diff(actual_doc, expected_doc):

    assert expected_doc["id"] == actual_doc["id"]
    assert expected_doc["crs"] == actual_doc["crs"]
    assert expected_doc["product"]["name"] == actual_doc["product"]["name"]
    assert expected_doc["label"] == actual_doc["label"]

    # Test geometry field
    doc_diff = deep_diff(expected_doc["geometry"], actual_doc["geometry"])
    assert doc_diff == {}, pformat(doc_diff)

    # Test grids field
    doc_diff = deep_diff(expected_doc["grids"], actual_doc["grids"])
    assert doc_diff == {}, pformat(doc_diff)

    # Test measurements field
    doc_diff = deep_diff(expected_doc["measurements"], actual_doc["measurements"])
    assert doc_diff == {}, pformat(doc_diff)

    # Test properties field
    doc_diff = deep_diff(
        expected_doc["properties"],
        actual_doc["properties"],
        exclude_paths=[
            "root['odc:product']",
            "root['proj:epsg']",
            "root['proj:shape']",
            "root['proj:transform']",
        ],
    )
    assert doc_diff == {}, pformat(doc_diff)

    # Test lineage field
    doc_diff = deep_diff(expected_doc["lineage"], actual_doc["lineage"])
    if expected_doc.get("accessories") is not None:
        doc_diff = deep_diff(expected_doc["accessories"], actual_doc["accessories"])
    assert doc_diff == {}, pformat(doc_diff)
