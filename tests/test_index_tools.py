import pytest
from datacube.utils import geometry as geom

from odc.index._grouper import group_by_nothing, key2num, mid_longitude, solar_offset


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
