"""Tests for UTM utilities in odc.index._utm."""

import pytest

from odc.index._utm import mk_utm_gs, utm_region_code, utm_tile_dss, utm_zone_to_epsg


@pytest.mark.parametrize(
    "zone,epsg",
    [("56S", 32756), ("55N", 32655), ("01S", 32701), ("01N", 32601), ("60S", 32760)],
)
def test_zone_to_epsg(zone, epsg):
    assert utm_zone_to_epsg(zone) == epsg
    assert utm_region_code(epsg) == zone
    assert utm_region_code(epsg, (1, 3)) == f"{zone}_01_03"
    assert utm_region_code((epsg, 1, 4)) == f"{zone}_01_04"


@pytest.mark.parametrize("zone", ["", "10", "61S", "100N", "XXS"])
def test_utm_unhappy_paths(zone):
    with pytest.raises(ValueError):
        utm_zone_to_epsg(zone)


@pytest.mark.parametrize("epsg", [4326, 3857, 32661, 32761])
def test_utm_region_code_unhappy_paths(epsg):
    with pytest.raises(ValueError):
        utm_region_code(epsg)


def test_mk_utm_gs():
    gs = mk_utm_gs(32756, (-10, 10), 297)
    assert gs.resolution == (-10, 10)
    assert gs.crs.epsg == 32756
    assert gs.tile_geobox((0, 0)).shape == (297, 297)
    assert gs.tile_size == (2970, 2970)

    gs = mk_utm_gs(32656, 10, 297)
    assert gs.resolution == (-10, 10)
    assert gs.crs.epsg == 32656
    assert gs.tile_geobox((0, 0)).shape == (297, 297)
    assert gs.tile_size == (2970, 2970)


def test_tile_dss_smoke_test(s2_dataset):
    tiles = utm_tile_dss([s2_dataset] * 3, resolution=10, pixels_per_cell=2000)
    assert len(tiles) > 0

    for tile in tiles:
        assert len(tile.region) == 3
        assert tile.region[0] == s2_dataset.crs.epsg
        assert len(tile.dss) == 3
        assert tile.grid_spec.tile_geobox(tile.region[1:]) == tile.geobox
        for ds in tile.dss:
            assert ds.extent.intersects(tile.geobox.extent)
