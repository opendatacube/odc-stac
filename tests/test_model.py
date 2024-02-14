# pylint: disable=redefined-outer-name,missing-module-docstring,missing-function-docstring
import datetime as dt

import pytest
from dask.base import tokenize
from odc.geo.geobox import GeoBox

from odc.stac import (
    ParsedItem,
    RasterBandMetadata,
    RasterCollectionMetadata,
    RasterLoadParams,
    RasterSource,
)
from odc.stac.testing.stac import b_, mk_parsed_item


def test_band_load_info():
    meta = RasterBandMetadata(data_type="uint16", nodata=13)
    band = RasterSource("https://example.com/some.tif", meta=meta)
    assert RasterLoadParams.same_as(meta).dtype == "uint16"
    assert RasterLoadParams.same_as(band).fill_value == 13

    band = RasterSource("file:///")
    assert RasterLoadParams.same_as(band).dtype == "float32"
    assert RasterLoadParams().dtype is None
    assert RasterLoadParams().nearest is True
    assert RasterLoadParams(resampling="average").nearest is False


@pytest.mark.parametrize("lon", [0, -179, 179, 10, 23.4])
def test_mid_longitude(lon: float):
    gbox = GeoBox.from_bbox((lon - 0.1, 0, lon + 0.1, 1), shape=(100, 100))
    xx = mk_parsed_item([b_("b1", gbox)])
    assert xx.geometry is not None
    assert xx.geometry.crs == "epsg:4326"
    assert xx.mid_longitude == pytest.approx(lon)

    assert mk_parsed_item([]).mid_longitude is None


def test_solar_day():
    def _mk(lon: float, datetime):
        gbox = GeoBox.from_bbox((lon - 0.1, 0, lon + 0.1, 1), shape=(100, 100))
        return mk_parsed_item([b_("b1", gbox)], datetime=datetime)

    for lon in [0, 1, 2, 3, 14, -1, -14, -3]:
        xx = _mk(lon, "2020-01-02T12:13:14Z")
        assert xx.mid_longitude == pytest.approx(lon)
        assert xx.nominal_datetime == xx.solar_date

    xx = _mk(15.1, "2020-01-02T12:13:14Z")
    assert xx.nominal_datetime != xx.solar_date
    assert xx.nominal_datetime + dt.timedelta(seconds=3600) == xx.solar_date
    assert xx.nominal_datetime + dt.timedelta(seconds=3600) == xx.solar_date_at(20)

    xx = _mk(-15.1, "2020-01-02T12:13:14Z")
    assert xx.nominal_datetime != xx.solar_date
    assert xx.nominal_datetime - dt.timedelta(seconds=3600) == xx.solar_date
    assert xx.nominal_datetime - dt.timedelta(seconds=3600) == xx.solar_date_at(-20)

    xx = mk_parsed_item([b_("b1")], datetime="2000-01-02")
    assert xx.geometry is None
    assert xx.nominal_datetime == xx.solar_date

    xx = _mk(10, None)
    with pytest.raises(ValueError):
        _ = xx.solar_date


@pytest.fixture()
def collection_ab() -> RasterCollectionMetadata:
    return RasterCollectionMetadata(
        "ab",
        {
            ("a", 1): RasterBandMetadata("uint8"),
            ("b", 1): RasterBandMetadata("uint16"),
        },
        {"A": [("a", 1)], "AA": [("a", 1)], "B": [("b", 1)]},
        has_proj=True,
        band2grid={},
    )


@pytest.fixture()
def parsed_item_ab(collection_ab: RasterCollectionMetadata) -> ParsedItem:
    return ParsedItem(
        "item-ab",
        collection_ab,
        {
            k: RasterSource(f"file:///{k[0]}-{k[1]}.tif", meta=collection_ab[k])
            for k in collection_ab
        },
    )


def test_collection(collection_ab: RasterCollectionMetadata):
    xx = collection_ab

    assert xx.canonical_name("b") == "b"
    assert xx.canonical_name("B") == "b"
    assert xx.canonical_name("AA") == "a"
    assert xx.canonical_name("a") == "a"

    assert xx.band_key("B") == ("b", 1)
    assert xx.band_key("AA") == ("a", 1)
    assert xx["AA"].data_type == "uint8"
    assert xx["b"].data_type == "uint16"
    assert "b" in xx
    assert "b.1" in xx
    assert ("b", 1) in xx
    assert {} not in xx
    assert ("some-random", 1) not in xx
    assert "no-such-band" not in xx

    assert xx.resolve_bands("AA")["AA"] == xx["a"]
    assert list(xx.resolve_bands(["a", "B"])) == ["a", "B"]
    assert xx.resolve_bands(["a", "B"])["B"] is xx["b"]
    assert xx.resolve_bands(["a", "B"])["a"] is xx["a"]
    assert set(xx) == set([("a", 1), ("b", 1)])
    assert len(xx) == 2

    for k in "a AA A b B".split(" "):
        assert xx.band_key(k) in xx.bands
        assert xx.canonical_name(k) in ["a", "b"]
        assert k in xx
        assert isinstance(xx[k], RasterBandMetadata)
        assert xx[k] is xx[xx.band_key(k)]

    with pytest.raises(ValueError):
        _ = xx.resolve_bands(["xxxxxxxx", "a"])

    with pytest.raises(KeyError):
        _ = xx["no-such-band"]


def test_collection_allbands():
    xx = mk_parsed_item([b_("a.1"), b_("a.2"), b_("a.3")])
    md = xx.collection
    assert md.all_bands == ["a.1", "a.2", "a.3"]

    md.aliases["AA"] = [("a", 2)]
    md.aliases["AAA"] = [("a", 3)]
    assert md["AA"] == md["a.2"]
    assert md["AAA"] == md["a.3"]

    # expect aliases to be used for all_band when multi-band
    # assets have unique aliases
    assert md.all_bands == ["a.1", "AA", "AAA"]
    assert md.canonical_name("a.2") == "AA"
    assert md.canonical_name("AA") == "AA"
    assert md.canonical_name("a.3") == "AAA"
    assert md.canonical_name("AAA") == "AAA"


def test_parsed_item(parsed_item_ab: ParsedItem):
    xx = parsed_item_ab
    assert xx["AA"] is not None
    assert xx["b"] is not None
    assert xx["AA"].meta is not None
    assert xx["AA"].meta.data_type == "uint8"
    assert xx["b"].meta is not None
    assert xx["b"].meta.data_type == "uint16"

    assert xx.resolve_bands("AA")["AA"] == xx["a"]
    assert list(xx.resolve_bands(["a", "B"])) == ["a", "B"]
    assert xx.resolve_bands(["a", "B"])["B"] is xx["b"]
    assert xx.resolve_bands(["a", "B"])["a"] is xx["a"]
    assert set(xx) == set([("a", 1), ("b", 1)])
    assert len(xx) == 2
    assert len(set([xx, xx, xx])) == 1
    assert ("a", 1) in xx
    assert ("a", 2) not in xx
    assert ("a", 2, 3) not in xx

    for k in "a AA A b B".split(" "):
        assert k in xx
        assert [k] not in xx
        assert f"___{k}___" not in xx
        assert isinstance(xx[k], RasterSource)
        assert xx[k] is xx.resolve_bands(k)[k]

    assert xx["b"].strip().geobox is None
    assert xx["b"].strip().meta is xx["b"].meta
    assert xx["b"].strip().uri == xx["b"].uri
    assert xx["b"].strip().band == xx["b"].band
    assert xx["b"].strip().subdataset == xx["b"].subdataset
    assert xx["b"].strip().driver_data == xx["b"].driver_data

    assert xx.strip()["b"].geobox is None
    assert xx.strip()["b"].meta is xx["b"].meta
    assert xx.strip()["b"].uri == xx["b"].uri
    assert xx.strip()["b"].band == xx["b"].band
    assert xx.strip()["b"].subdataset == xx["b"].subdataset
    assert xx.strip()["b"].driver_data == xx["b"].driver_data


def test_tokenize(parsed_item_ab: ParsedItem):
    assert tokenize(parsed_item_ab.collection) == tokenize(parsed_item_ab.collection)
    assert tokenize(parsed_item_ab) == tokenize(parsed_item_ab)
    assert tokenize(parsed_item_ab["a"]) == tokenize(parsed_item_ab["a"])
    assert tokenize(parsed_item_ab["a"].meta) == tokenize(parsed_item_ab["a"].meta)

    assert tokenize(RasterLoadParams()) == tokenize(RasterLoadParams())
    assert tokenize(RasterLoadParams("uint8")) == tokenize(RasterLoadParams("uint8"))
    assert tokenize(RasterLoadParams("uint8")) != tokenize(RasterLoadParams("uint32"))
