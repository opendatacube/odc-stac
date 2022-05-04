"""
Making STAC items for testing.
"""
from datetime import datetime, timezone

import pystac.asset
import pystac.item
import xarray as xr
from odc.geo.geobox import GeoBox
from pystac.extensions.projection import ProjectionExtension
from pystac.extensions.raster import RasterBand, RasterExtension
from toolz import dicttoolz

from .._mdtools import _group_geoboxes
from .._model import (
    ParsedItem,
    RasterBandMetadata,
    RasterCollectionMetadata,
    RasterSource,
)

# pylint: disable=redefined-builtin,too-many-arguments

STAC_DATE_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"
STAC_DATE_FMT_SHORT = "%Y-%m-%dT%H:%M:%SZ"


def _norm_dates(*args):
    valid = [a for a in args if a is not None]
    valid = [
        datetime.fromisoformat(dt).replace(tzinfo=timezone.utc)
        for dt in xr.DataArray(list(valid))
        .astype("datetime64[ns]")
        .dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
        .values
    ]
    valid = iter(valid)
    return [next(valid) if a else None for a in args]


def b_(
    name,
    geobox=None,
    dtype="int16",
    nodata=None,
    unit="1",
    uri=None,
    bidx=1,
    prefix="http://example.com/items/",
):
    if uri is None:
        uri = f"{prefix}{name}.tif"
    meta = RasterBandMetadata(dtype, nodata, unit)
    return (name, RasterSource(uri, bidx, geobox=geobox, meta=meta))


def mk_parsed_item(
    bands,
    datetime=None,
    start_datetime=None,
    end_datetime=None,
    id="some-item",
    collection="some-collection",
    href=None,
) -> ParsedItem:
    """
    Construct parsed stac item for testing.
    """
    # pylint: disable=redefined-outer-name
    if isinstance(bands, (list, tuple)):
        bands = dict(bands)

    gboxes = dicttoolz.valmap(lambda b: b.geobox, bands)
    gboxes = dicttoolz.valfilter(lambda x: x is not None, gboxes)

    if len(gboxes) == 0:
        band2grid = {b: "default" for b in bands}
        geobox = None
    else:
        grids, band2grid = _group_geoboxes(gboxes)
        geobox = grids["default"]

    if geobox is not None:
        geometry = geobox.geographic_extent
    else:
        geometry = None

    collection = RasterCollectionMetadata(
        collection,
        dicttoolz.valmap(lambda b: b.meta, bands),
        aliases={},
        has_proj=(geobox is not None),
        band2grid=band2grid,
    )
    datetime, start_datetime, end_datetime = _norm_dates(
        datetime, start_datetime, end_datetime
    )

    return ParsedItem(
        id,
        collection,
        bands,
        geometry=geometry,
        datetime=datetime,
        datetime_range=(start_datetime, end_datetime),
        href=href,
    )


def _add_proj(gbox: GeoBox, xx):
    proj = ProjectionExtension.ext(xx, add_if_missing=True)
    proj.shape = list(gbox.shape.yx)
    proj.transform = gbox.transform[:6]
    crs = gbox.crs
    if crs is not None:
        epsg = crs.epsg
        if epsg is not None:
            proj.epsg = epsg
        else:
            proj.wkt2 = crs.wkt


def to_stac_item(item: ParsedItem) -> pystac.item.Item:
    gg = item.geometry

    props = {}
    for n, dt in zip(["start_datetime", "end_datetime"], item.datetime_range):
        if dt is not None:
            props[n] = dt.strftime(STAC_DATE_FMT)

    xx = pystac.item.Item(
        item.id,
        geometry=gg.json if gg is not None else None,
        bbox=list(gg.boundingbox.bbox) if gg is not None else None,
        datetime=item.datetime,
        properties=props,
        collection=item.collection.name,
    )

    RasterExtension.add_to(xx)
    gboxes = item.geoboxes()
    if len(gboxes) > 0:
        gbox = gboxes[0]

        ProjectionExtension.add_to(xx)
        _add_proj(gbox, xx)

    for asset_name, b in item.bands.items():
        xx.add_asset(
            asset_name,
            pystac.asset.Asset(b.uri, media_type="image/tiff", roles=["data"]),
        )

    for asset_name, asset in xx.assets.items():
        bb = item.bands[asset_name]
        meta = bb.meta
        assert meta is not None

        RasterExtension(asset).apply(
            [
                RasterBand.create(
                    data_type=meta.data_type,  # type: ignore
                    nodata=meta.nodata,
                    unit=meta.unit,
                )
            ]
        )

        if bb.geobox is not None:
            _add_proj(bb.geobox, asset)

    if item.href is not None:
        xx.set_self_href(item.href)

    return xx
