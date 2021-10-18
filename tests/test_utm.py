"""Tests for UTM utilities in odc.index._utm."""

import pytest

from datacube.utils.geometry import unary_union
from odc.index import bin_dataset_stream, bin_dataset_stream2
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


def test_bin_dataset_stream(s2_dataset):
    gridspec = mk_utm_gs(s2_dataset.crs.epsg, 10, 2000)
    dss = iter([s2_dataset] * 3)
    cells = {}

    _dss = bin_dataset_stream(gridspec, dss, cells)
    assert len(cells) == 0

    for ds in _dss:
        assert ds is s2_dataset

    assert len(cells) > 0
    extents = []

    for idx, cell in cells.items():
        assert len(cell.dss) == 3
        assert idx == cell.idx
        assert gridspec.tile_geobox(cell.idx) == cell.geobox
        extents.append(cell.geobox.extent)
        for _id in cell.dss:
            assert _id == s2_dataset.id
            assert s2_dataset.extent.intersects(cell.geobox.extent)

    # No dataset parts outside of occupied tiles
    tiles_extent = unary_union(extents)
    assert tiles_extent is not None
    assert tiles_extent.contains(s2_dataset.extent)


def test_bin_dataset_stream2(s2_dataset):
    gridspec = mk_utm_gs(s2_dataset.crs.epsg, 10, 2000)
    dss = iter([s2_dataset] * 3)

    for ds, tiles in bin_dataset_stream2(gridspec, dss):
        assert ds is s2_dataset
        extents = []
        for tidx in tiles:
            geobox = gridspec.tile_geobox(tidx)
            assert ds.extent.intersects(geobox.extent)
            extents.append(geobox.extent)

        # Verify that union of all tiles encloses dataset
        tiles_extent = unary_union(extents)
        assert tiles_extent is not None
        assert tiles_extent.contains(ds.extent)
