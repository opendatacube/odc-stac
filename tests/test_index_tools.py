import datetime
import json
from unittest.mock import MagicMock

import pytest
import yaml
from datacube.utils import geometry as geom

from odc.index._grouper import group_by_nothing, key2num, mid_longitude, solar_offset
from odc.index._index import (
    month_range,
    parse_doc_stream,
    product_from_yaml,
    season_range,
    time_range,
)


@pytest.mark.parametrize("lon,lat", [(0, 10), (100, -10), (-120, 30)])
def test_mid_lon(lon, lat):
    r = 0.1
    rect = geom.box(lon - r, lat - r, lon + r, lat + r, "epsg:4326")
    assert rect.centroid.coords[0] == pytest.approx((lon, lat))

    assert mid_longitude(rect) == pytest.approx(lon)
    assert mid_longitude(rect.to_crs("epsg:3857")) == pytest.approx(lon)

    offset = solar_offset(rect, "h")
    assert offset.seconds % (60 * 60) == 0

    offset_sec = solar_offset(rect, "s")
    assert abs((offset - offset_sec).seconds) <= 60 * 60


@pytest.mark.parametrize(
    "input,expect",
    [
        ("ABAAC", [0, 1, 0, 0, 2]),
        ("B", [0]),
        ([1, 1, 1], [0, 0, 0]),
        ("ABCC", [0, 1, 2, 2]),
    ],
)
def test_key2num(input, expect):
    rr = list(key2num(input))
    assert rr == expect

    reverse = {}
    rr = list(key2num(input, reverse))
    assert rr == expect
    assert set(reverse.keys()) == set(range(len(set(input))))
    assert set(reverse.values()) == set(input)
    # first entry always gets an index of 0
    assert reverse[0] == input[0]


def test_grouper(s2_dataset):
    xx = group_by_nothing([s2_dataset])
    assert xx.values[0] == (s2_dataset,)
    assert xx.uuid.values[0] == s2_dataset.id

    xx = group_by_nothing([s2_dataset, s2_dataset], solar_offset(s2_dataset.extent))
    assert xx.values[0] == (s2_dataset,)
    assert xx.values[0] == (s2_dataset,)
    assert xx.uuid.values[1] == s2_dataset.id
    assert xx.uuid.values[1] == s2_dataset.id


def test_month_range():
    m1, m2 = month_range(2019, 1, 3)
    assert m1.year == 2019
    assert m2.year == 2019
    assert m1.month == 1 and m2.month == 3

    m1, m2 = month_range(2019, 12, 3)
    assert m1 == datetime.datetime(2019, 12, 1)
    assert m2 == datetime.datetime(2020, 2, 29, 23, 59, 59, 999999)

    assert month_range(2018, 12, 4) == month_range(2019, -1, 4)

    assert season_range(2019, "djf") == month_range(2019, -1, 3)
    assert season_range(2019, "mam") == month_range(2019, 3, 3)
    assert season_range(2002, "jja") == month_range(2002, 6, 3)
    assert season_range(2000, "son") == month_range(2000, 9, 3)

    with pytest.raises(ValueError):
        season_range(2000, "ham")


@pytest.mark.parametrize(
    "t1, t2",
    [
        ("2020-01-17", "2020-03-19"),
        ("2020-04-17", "2020-04-19"),
        ("2020-02-01T13:08:01", "2020-09-03T00:33:46.103"),
        ("2000-01-23", "2001-01-19"),
    ],
)
def test_time_range(t1, t2):
    t1, t2 = map(datetime.datetime.fromisoformat, (t1, t2))
    tt = list(time_range(t1, t2))

    assert tt[0][0] == t1
    assert tt[-1][-1] == t2

    if len(tt) == 1:
        assert tt[0] == (t1, t2)

    t_prev = tt[0][-1]
    for t1, t2 in tt[1:]:
        assert t1 > t_prev
        assert (t1 - t_prev).seconds < 1
        t_prev = t2


def test_parse_doc_stream():
    sample_data = {"a": 10, "b": [1]}
    _json = json.dumps(sample_data)
    _yaml = yaml.dump(sample_data)
    data = [
        ("http://example.com/foo.json", _json),
        ("file:///blah.yml", _yaml),
        ("s3://bu/f.json", "!!not a valid json!!"),
    ]
    _docs = list(parse_doc_stream(iter(data)))
    assert len(_docs) == 3
    assert _docs[0] == (data[0][0], sample_data)
    assert _docs[1] == (data[1][0], sample_data)
    assert _docs[2] == (data[2][0], None)

    on_error = MagicMock()
    transform = lambda d: dict(**d, seen=True)
    _docs = list(parse_doc_stream(iter(data), transform=transform, on_error=on_error))
    assert len(_docs) == 3
    assert _docs[0][1] is not None
    assert _docs[1][1] is not None
    assert _docs[2][1] is None
    assert _docs[0][1]["seen"]
    assert _docs[1][1]["seen"]
    assert on_error.call_count == 1


def test_product_from_yaml(test_data_dir):
    p = product_from_yaml(str(test_data_dir / "test-product-eo3.yml"))
    assert p.name == "test_product_eo3"
    assert p.metadata_type.name == "eo3"

    p = product_from_yaml(str(test_data_dir / "test-product-eo.yml"))
    assert p.name == "test_product_eo"
    assert p.metadata_type.name == "eo"
