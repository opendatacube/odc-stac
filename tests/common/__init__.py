from pystac import Item


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
            "properties": {"datetime": datetime, **props,},
            "geometry": geometry,
            "links": [],
            "assets": {},
            "stac_extensions": stac_extensions,
        }
    )
