# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: ODC
#     language: python
#     name: odc
# ---

# %%
import json
from timeit import default_timer as t_now

import geopandas as gpd
import numpy as np
import odc.stac
import planetary_computer as pc
import pystac.item
import pystac_client
from dask.utils import format_bytes
from distributed import Client
from distributed import wait as dask_wait

if "geom_query" in locals():
    bbox = tuple(geom_query.boundingbox)

mode = "site1-tall"
if mode == "site1-tall":
    # mgrs_tiles = ["35MNM", "35LNL", "35LNK", "35LNJ", "35LNH", "35LNG", "35LNF", "35LNE", "35LND"]
    bbox = (27.345815, -14.98724, 27.565542, -7.710992)  # Narrow/Tall epsg:32735
    file_id = "site1-20200606-tall-strip-africa"
    datetime = "2020-06-06"
    query = {}
elif mode == "site2":
    bbox = None
    file_id = "site2-2020_jun_jul-35MNM"
    datetime = "2020-06/2020-07"
    query = {
        "s2:mgrs_tile": {"eq": "35MNM"},
        "s2:nodata_pixel_percentage": {"lt": 10},
    }


cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
search = cat.search(
    collections=["sentinel-2-l2a"],
    datetime=datetime,
    query=query,
    bbox=bbox,
)
print("Query API end-point")
all_features = search.item_collection_as_dict()

all_features["properties"] = dict(url=search.url, query=search._parameters)
all_features["properties"]

# %%
out_path = Path(f"{file_id}.geojson")
if out_path.exists():
    print(f"File exists, keeping previous version: {out_path}")
else:
    print(f"Writing to: {out_path}")
    json.dump(all_features, open(out_path, "wt"))

# %%
all_items = [pystac.item.Item.from_dict(f) for f in all_features["features"]]

# %%
gdf = gpd.GeoDataFrame.from_features(all_features, "epsg:4326")
display(set(gdf["s2:mgrs_tile"].values), set(gdf.platform), len(set(gdf.datetime)))

_map = gdf.explore(
    "s2:mgrs_tile",
    categorical=True,
    tooltip=[
        "s2:mgrs_tile",
        "datetime",
        "s2:nodata_pixel_percentage",
        "eo:cloud_cover",
    ],
    popup=True,
    style_kwds=dict(fillOpacity=0.0, width=2),
    name="STAC",
)
display(_map)

# %%
display(gdf.head())
# gdf[gdf['s2:nodata_pixel_percentage']>10].explore()

# %%
