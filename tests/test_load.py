from copy import deepcopy
from unittest.mock import MagicMock

import pystac
import pystac.item
import pytest
import shapely.geometry
from odc.geo.xr import ODCExtension

from odc.stac._load import load as stac_load


def test_stac_load_smoketest(sentinel_stac_ms_with_raster_ext: pystac.item.Item):
    item = sentinel_stac_ms_with_raster_ext.clone()

    params = dict(crs="EPSG:3857", resolution=100, align=0, chunks={})
    with pytest.warns(UserWarning, match="`rededge`"):
        xx = stac_load([item], "B02", **params)

    assert isinstance(xx.B02.odc, ODCExtension)
    assert xx.B02.shape[0] == 1
    assert xx.B02.odc.geobox.crs == "EPSG:3857"

    # Test dc.load name for bands, and alias support
    with pytest.warns(UserWarning, match="`rededge`"):
        xx = stac_load([item], measurements=["red", "green"], **params)

    assert "red" in xx.data_vars
    assert "green" in xx.data_vars
    assert xx.red.shape == xx.green.shape

    # Test dc.load name for bands, and alias support
    patch_url = MagicMock(return_value="https://example.com/f.tif")
    xx = stac_load(
        [item],
        measurements=["red", "green"],
        patch_url=patch_url,
        stac_cfg={"*": {"warnings": "ignore"}},
        **params,
    )
    assert isinstance(xx.odc, ODCExtension)

    # expect patch_url to be called 2 times, 1 for red and 1 for green band
    assert patch_url.call_count == 2

    patch_url = MagicMock(return_value="https://example.com/f.tif")
    zz = stac_load(
        [item],
        patch_url=patch_url,
        stac_cfg={"*": {"warnings": "ignore"}},
        **params,
    )
    assert patch_url.call_count == len(zz.data_vars)

    yy = stac_load(
        [item], ["nir"], like=xx, chunks={}, stac_cfg={"*": {"warnings": "ignore"}}
    )
    assert yy.odc.geobox == xx.odc.geobox

    yy = stac_load(
        [item],
        ["nir"],
        geobox=xx.odc.geobox,
        chunks={},
        stac_cfg={"*": {"warnings": "ignore"}},
    )
    assert yy.odc.geobox == xx.odc.geobox
    assert yy.odc.geobox == yy.nir.odc.geobox

    # Check automatic CRS/resolution
    yy = stac_load(
        [item],
        ["nir", "coastal"],
        chunks={},
        stac_cfg={"*": {"warnings": "ignore"}},
    )
    assert yy.odc.geobox.crs == "EPSG:32606"
    assert yy.odc.geobox.resolution.yx == (-10, 10)

    # test bbox overlaping with lon/lat
    with pytest.raises(ValueError):
        stac_load([item], ["nir"], bbox=(0, 0, 1, 1), lon=(0, 1), lat=(0, 1), chunks={})

    # test bbox overlaping with x/y
    with pytest.raises(ValueError):
        stac_load(
            [item],
            ["nir"],
            bbox=(0, 0, 1, 1),
            x=(0, 1000),
            y=(0, 1000),
            chunks={},
        )

    bbox = (0, 0, 1, 1)
    x1, y1, x2, y2 = bbox

    assert (
        stac_load(
            [item],
            ["nir"],
            crs="epsg:3857",
            resolution=10,
            chunks={},
            lon=(x1, x2),
            lat=(y1, y2),
        ).nir.odc.geobox
        == stac_load(
            [item],
            ["nir"],
            crs="epsg:3857",
            resolution=10,
            chunks={},
            bbox=bbox,
        ).nir.odc.geobox
    )

    geopolygon = shapely.geometry.box(*bbox)
    assert (
        stac_load(
            [item],
            ["nir"],
            crs="epsg:3857",
            resolution=10,
            chunks={},
            lon=(x1, x2),
            lat=(y1, y2),
        ).nir.odc.geobox
        == stac_load(
            [item],
            ["nir"],
            crs="epsg:3857",
            resolution=10,
            chunks={},
            geopolygon=geopolygon,
        ).nir.odc.geobox
    )
