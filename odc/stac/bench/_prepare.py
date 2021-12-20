"""Utilities for benchmarking."""
import json
from pathlib import Path
from typing import Any, Dict

# pylint: disable=import-outside-toplevel

SAMPLE_SITES = {
    "s2-ms-mosaic": {
        "file_id": "s2-ms-mosaic_2020-06-06--P1D",
        "api": "https://planetarycomputer.microsoft.com/api/stac/v1",
        "search": {
            "collections": ["sentinel-2-l2a"],
            "datetime": "2020-06-06",
            "bbox": [27.345815, -14.98724, 27.565542, -7.710992],
            "query": {},
        },
    },
    "s2-ms-deep": {
        "file_id": "s2-ms-deep_2020-06--P2M_35MNM",
        "api": "https://planetarycomputer.microsoft.com/api/stac/v1",
        "search": {
            "collections": ["sentinel-2-l2a"],
            "datetime": "2020-06/2020-07",
            "bbox": None,
            "query": {
                "s2:mgrs_tile": {"eq": "35MNM"},
                "s2:nodata_pixel_percentage": {"lt": 10},
            },
        },
    },
}


def dump_site(site: Dict[str, Any], overwrite: bool = False) -> Dict[str, Any]:
    """
    Prepare input for benchmarking.

    Queries API end-point according to site configuration and dumps result into a geojson file. Site
    configuration must include ``file_id:str, api:str, search:Dict[str,Any]``.

    .. code-block:: json

       {
         "file_id": "ms-s2-long-mosaic_2020-06-06--P1D",
         "api": "https://planetarycomputer.microsoft.com/api/stac/v1",
         "search": {
           "collections": ["sentinel-2-l2a"],
           "datetime": "2020-06-06",
           "bbox": [ 27.345815, -14.98724, 27.565542, -7.710992],
           "query": {}
         }
       }

    :param site: Definition of the test query
    :param overwrite: overwrite existing file
    :return: Returns GeoJSON FeatureCollection with extra metadata about the query
    """
    import pystac_client

    api = site["api"]
    search = site["search"]

    cat = pystac_client.Client.open(api)
    search = cat.search(**search)
    print(f"Query API end-point: {api}")
    all_features = search.get_all_items_as_dict()
    all_features["properties"] = dict(
        api=search.url, search=search._parameters  # pylint: disable=protected-access
    )

    out_path = Path(f"{site['file_id']}.geojson")
    if out_path.exists():
        if overwrite:
            print(f"Will overwrite: {out_path}")
        else:
            print(f"File exists, keeping previous version: {out_path}")
            return all_features

    print(f"Writing to: {out_path}")
    with open(out_path, "wt", encoding="utf8") as dst:
        json.dump(all_features, dst)

    return all_features
