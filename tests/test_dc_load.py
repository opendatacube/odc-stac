from copy import deepcopy
from unittest.mock import MagicMock

import geopandas as gpd
import geopandas.datasets
import pystac
import pystac.item
import pytest
import shapely.geometry
from datacube.model import Dataset
from pyproj.crs.crs import CRS
from shapely.geometry import geo

from odc.stac import configure_rio, dc_load, eo3_geoboxes, stac2ds, stac_load
from odc.stac._dcload import _geojson_to_shapely, _normalize_geometry
from odc.stac._load import most_common_crs


def test_dc_load_smoketest(sentinel_stac_ms: pystac.item.Item):
    item = sentinel_stac_ms
    with pytest.warns(UserWarning, match="`rededge`"):
        (ds,) = stac2ds([item], {})

    params = dict(output_crs=ds.crs, resolution=(-100, 100), chunks={})
    xx = dc_load([ds], "B02", **params)
    assert xx.B02.shape == (1, 1099, 1099)
    assert xx.B02.geobox.crs == ds.crs
    assert "units" not in xx.time.attrs

    # Check that aliases also work
    xx = dc_load([ds], bands=["red", "green", "blue"], **params)
    assert xx.green.shape == xx.red.shape
    assert xx.blue.dtype == xx.red.dtype

    # Check that dask_chunks= is an alias for chunks=
    params["dask_chunks"] = params.pop("chunks")
    xx = dc_load([ds], "nir", **params)
    assert xx.nir.chunks == tuple((s,) for s in xx.nir.shape)

    with pytest.warns(UserWarning, match="Supplied 'geobox=' parameter aliases"):
        yy = dc_load([ds], "SCL", geobox=xx.nir.geobox, **params)
    assert xx.nir.geobox == yy.SCL.geobox


def test_stac_load_smoketest(sentinel_stac_ms_with_raster_ext: pystac.item.Item):
    item = sentinel_stac_ms_with_raster_ext.clone()

    params = dict(crs="EPSG:3857", resolution=100, align=0, chunks={})
    with pytest.warns(UserWarning, match="`rededge`"):
        xx = stac_load([item], "B02", **params)

    assert xx.B02.shape[0] == 1
    assert xx.B02.geobox.crs == "EPSG:3857"

    # Test dc.load name for bands, and alias support
    with pytest.warns(UserWarning, match="`rededge`"):
        xx = stac_load([item], measurements=["red", "green"], **params)

    assert "red" in xx.data_vars
    assert "green" in xx.data_vars
    assert xx.red.shape == xx.green.shape

    # Test dc.load name for bands, and alias support
    patch_url = MagicMock(return_value="https://example.com/f.tif")
    product_cache = {}
    xx = stac_load(
        [item],
        measurements=["red", "green"],
        patch_url=patch_url,
        stac_cfg={"*": {"warnings": "ignore"}},
        product_cache=product_cache,
        **params,
    )
    # expect patch_url to be called 2 times, 1 for red and 1 for green band
    assert patch_url.call_count == 2
    # expect product cache to contain 1 product
    assert len(product_cache) == 1

    patch_url = MagicMock(return_value="https://example.com/f.tif")
    zz = stac_load(
        [item],
        patch_url=patch_url,
        stac_cfg={"*": {"warnings": "ignore"}},
        product_cache=product_cache,
        **params,
    )
    assert patch_url.call_count == len(zz.data_vars)

    yy = stac_load(
        [item], ["nir"], like=xx, chunks={}, stac_cfg={"*": {"warnings": "ignore"}}
    )
    assert yy.nir.geobox == xx.geobox

    yy = stac_load(
        [item],
        ["nir"],
        geobox=xx.geobox,
        chunks={},
        stac_cfg={"*": {"warnings": "ignore"}},
    )
    assert yy.nir.geobox == xx.geobox

    # Check automatic CRS/resolution
    yy = stac_load(
        [item],
        ["nir", "coastal"],
        chunks={},
        stac_cfg={"*": {"warnings": "ignore"}},
    )
    assert yy.nir.geobox.crs == CRS("EPSG:32606")
    assert yy.nir.geobox.resolution == (-10, 10)

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
            product_cache=product_cache,
        ).nir.geobox
        == stac_load(
            [item],
            ["nir"],
            crs="epsg:3857",
            resolution=10,
            chunks={},
            bbox=bbox,
            product_cache=product_cache,
        ).nir.geobox
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
            product_cache=product_cache,
        ).nir.geobox
        == stac_load(
            [item],
            ["nir"],
            crs="epsg:3857",
            resolution=10,
            chunks={},
            geopolygon=geopolygon,
            product_cache=product_cache,
        ).nir.geobox
    )


def test_eo3_geoboxes(s2_dataset):
    geoboxes = eo3_geoboxes(s2_dataset)
    assert len(geoboxes) == 3

    geoboxes = eo3_geoboxes(s2_dataset, grids=["default"])
    assert len(geoboxes) == 1
    assert list(geoboxes) == ["default"]
    assert geoboxes["default"].crs == s2_dataset.crs
    assert geoboxes["default"].resolution == (-10, 10)
    assert geoboxes["default"].alignment == (0, 0)

    geoboxes = eo3_geoboxes(s2_dataset, bands=["red", "B02"])
    assert len(geoboxes) == 1
    assert list(geoboxes) == ["default"]

    doc = deepcopy(s2_dataset.metadata_doc)
    doc.pop("grids")
    ds = Dataset(s2_dataset.type, doc, [])
    with pytest.raises(ValueError, match="Missing grids, .*"):
        eo3_geoboxes(ds)

    doc = deepcopy(s2_dataset.metadata_doc)
    doc["grids"]["default"].pop("shape")
    ds = Dataset(s2_dataset.type, doc, [])
    with pytest.raises(
        ValueError, match=r"Each grid must have \.shape and \.transform"
    ):
        eo3_geoboxes(ds)

    doc = deepcopy(s2_dataset.metadata_doc)
    doc["grids"]["default"]["shape"] = (1, 1, 3)
    ds = Dataset(s2_dataset.type, doc, [])
    with pytest.raises(ValueError, match="Shape must contain.*"):
        eo3_geoboxes(ds)

    doc = deepcopy(s2_dataset.metadata_doc)
    doc["grids"]["default"]["transform"] = [1, 2, 3]
    ds = Dataset(s2_dataset.type, doc, [])
    with pytest.raises(ValueError, match="Invalid `transform` specified, .*"):
        eo3_geoboxes(ds)


def test_most_common_crs():
    epsg4326 = CRS("epsg:4326")
    epsg3857 = CRS("epsg:3857")

    assert most_common_crs([epsg3857, epsg4326, epsg4326]) is epsg4326
    assert most_common_crs([epsg3857, epsg3857, epsg4326]) is epsg3857


def test_normalize_geometry(sample_geojson):
    # sample_geojson has one feature wrapped in featurecollection
    epsg4326 = CRS("epsg:4326")
    g0 = _normalize_geometry(sample_geojson)
    assert g0.crs == epsg4326
    assert _normalize_geometry(sample_geojson["features"][0]) == g0
    assert _normalize_geometry(sample_geojson["features"][0]["geometry"]) == g0

    assert _normalize_geometry(g0) is g0

    g_shapely = _geojson_to_shapely(sample_geojson)
    assert _normalize_geometry(g_shapely) == g0

    gg = gpd.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
    geom = _normalize_geometry(gg[gg.continent == "Africa"])
    assert geom.crs == epsg4326
    assert geom.contains(_normalize_geometry(gg[gg.name == "Tanzania"]))

    with pytest.raises(ValueError):
        _normalize_geometry({})

    # some object without __geo_interface__
    with pytest.raises(ValueError):
        _normalize_geometry(epsg4326)


def test_configure_rio(capsys):
    def fake_register(plugin, name="name"):
        worker = MagicMock()
        plugin.setup(worker)

    client = MagicMock()
    client.register_worker_plugin = fake_register

    configure_rio(cloud_defaults=True, activate=True, verbose=False)
    _io = capsys.readouterr()
    assert _io.out == ""
    assert _io.err == ""

    configure_rio(cloud_defaults=True, activate=True, verbose=True)
    _io = capsys.readouterr()
    assert "GDAL_DISABLE_READDIR_ON_OPEN" in _io.out

    configure_rio(cloud_defaults=True, activate=True, verbose=True, client=client)
    _io = capsys.readouterr()
    assert "GDAL_DISABLE_READDIR_ON_OPEN" in _io.out
