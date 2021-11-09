# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Access Sentinel 2 Data from AWS
#
# https://registry.opendata.aws/sentinel-2-l2a-cogs/

# %%
import odc.ui
import yaml
from ipyleaflet import FullScreenControl, GeoJSON, LayersControl, Map, Rectangle
from IPython.display import Image, display
from odc.algo import to_rgba
from odc.stac import stac_load
from pystac_client import Client

# %%
cfg = """---
sentinel-s2-l2a-cogs:
  assets:
    '*':
      data_type: uint16
      nodata: 0
      unit: '1'
    SCL:
      data_type: uint8
      nodata: 0
      unit: '1'
    visual:
      data_type: uint8
      nodata: 0
      unit: '1'
  aliases:  # Alias -> Canonical Name
    red: B04
    green: B03
    blue: B02
"*":
  warnings: ignore # Disable warnings about duplicate common names
"""
cfg = yaml.load(cfg, Loader=yaml.SafeLoader)

catalog = Client.open("https://earth-search.aws.element84.com/v0")

# %% [markdown]
# ## Find STAC Items to Load

# %%
km2deg = 1.0 / 111
x, y = (113.887, -25.843)  # Center point of a query
r = 100 * km2deg
bbox = (x - r, y - r, x + r, y + r)

query = catalog.search(
    collections=["sentinel-s2-l2a-cogs"], datetime="2021-09-16", limit=100, bbox=bbox
)

items = list(query.get_items())
print(f"Found: {len(items):d} datasets")

# %% [markdown]
# ## Plot STAC Items on a Map

# %%
query_rectangle = Rectangle(
    bounds=(
        bbox[:2][::-1],
        bbox[2:][::-1],
    ),  # IPyleaflet expects ((lat1, lon1), (lat2, lon2))
    fill=False,
    weight=3,
    opacity=0.7,
    color="olive",
    name="Query",
)

# Convert STAC items into a GeoJSON FeatureCollection
stac_json = {
    "type": "FeatureCollection",
    "features": [item.to_dict() for item in items],
}

footprint_style = dict(
    fillColor="black",
    fillOpacity=0.0,
    weight=1,
    opacity=0.6,
    color="magenta",
    dashArray=1,
)
hover_style = dict(
    weight=4,
    opacity=1,
    color="tomato",
)

# Make GeoJSON layer with styles
stac_layer = GeoJSON(
    data=stac_json,
    style=footprint_style,
    hover_style=hover_style,
    name="STAC",
)

# Make a leaflet.Map object
map1 = Map(scroll_wheel_zoom=True, center=(y, x), zoom=5)
map1.layout.height = "300px"
map1.layout.width = "300px"

# Plot query rectangle
map1.add_layer(query_rectangle)
# Plot footprints on top
map1.add_layer(stac_layer)

display(map1)

# %% [markdown]
# ## Construct Dask Dataset
#
# Note that even though there are 9 STAC Items on input, there is only one
# timeslice on output. This is because of `groupby="solar_day"`. With that
# setting `stac_load` will place all items that occured on the same day (as
# adjusted for the timezone) into one image plane.

# %%
# Since we will plot it on a map we need to use `EPSG:3857` projection
crs = "epsg:3857"
zoom = 2 ** 5  # overview level 5

xx = stac_load(
    items,
    bands=("red", "green", "blue"),
    crs=crs,
    resolution=10 * zoom,
    chunks={},  # <-- use Dask
    groupby="solar_day",
    stac_cfg=cfg,
)
display(xx)

# %% [markdown]
# ## Load data and convert to RGBA

# %%
# %%time
rgba = to_rgba(xx, clamp=(1, 3000))
_rgba = rgba.compute()

# %% [markdown]
# ## Display Image on a map

# %%
# This compresses image with png and packs it into `data` url
# it then computes Image bounds and return `ipyleaflet.ImageOverlay`
ovr = odc.ui.mk_image_overlay(_rgba)

# Make a leaflet.Map object
lon, lat = rgba.geobox.geographic_extent.centroid.coords[0]
map2 = Map(scroll_wheel_zoom=True, center=(lat, lon), zoom=8)
map2.layout.height = "600px"
map2.add_control(FullScreenControl())
map2.add_control(LayersControl())

# Add Image overlay
map2.add_layer(ovr)

# Plot footprints on top
map2.add_layer(query_rectangle)
map2.add_layer(stac_layer)

display(map2)

# %% [markdown]
# ## Load with bounding box
#
# As you can see `stac_load` returned all the data covered by STAC items
# returned from the query. This happens by default as `stac_load` has no way of
# knowing what your query was. But it is possible to control what region is
# loaded. There are several mechanisms available, but probably simplest one is
# to use `bbox=` parameter (compatible with `stac_client`).
#
# Let's load a small region at native resolution to demonstrate.

# %%
r = 6 * km2deg
small_bbox = (x - r, y - r, x + r, y + r)

yy = stac_load(
    items,
    bands=("red", "green", "blue"),
    crs=crs,
    resolution=10,
    chunks={},  # <-- use Dask
    groupby="solar_day",
    stac_cfg=cfg,
    bbox=small_bbox,
)
im_small = to_rgba(yy, clamp=(1, 3000)).compute()

# %%
display(Image(data=odc.ui.to_jpeg_data(im_small.isel(time=0).data, quality=80)))

# %% [markdown]
# --------------------------------------------------------------
