from pystac import Item

from odc.stac._mdtools import RasterBandMetadata

# fmt: off
S2_ALL_BANDS = {
    "B01", "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B09", "B11", "B12", "B8A",
    "AOT", "SCL", "WVP", "visual",
}
# fmt: on


STAC_CFG = {
    "sentinel-2-l2a": {
        "assets": {
            "*": RasterBandMetadata("uint16", 0, "1"),
            "SCL": RasterBandMetadata("uint8", 0, "1"),
            "visual": dict(data_type="uint8", nodata=0, unit="1"),
        },
        "aliases": {  # Work around duplicate rededge common_name
            "rededge": "B05",
            "rededge1": "B05",
            "rededge2": "B06",
            "rededge3": "B07",
        },
    }
}

NO_WARN_CFG = {"*": {"warnings": "ignore"}}


def mk_stac_item(
    _id, datetime="2012-12-12T00:00:00Z", geometry=None, stac_extensions=None, **props
):
    if stac_extensions is None:
        stac_extensions = []

    return Item.from_dict(
        {
            "type": "Feature",
            "stac_version": "1.0.0",
            "id": str(_id),
            "properties": {
                "datetime": datetime,
                **props,
            },
            "geometry": geometry,
            "links": [],
            "assets": {},
            "stac_extensions": stac_extensions,
        }
    )
