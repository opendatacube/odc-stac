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
from IPython.display import display
from odc.algo import to_rgba
from pystac_client import Client

from odc.stac import stac_load

# %%
cfg = """---
"*":
  warnings: ignore # Disable warnings about duplicate common names
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
"""
cfg = yaml.load(cfg, Loader=yaml.SafeLoader)

catalog = Client.open("https://earth-search.aws.element84.com/v0")

# %% [markdown]
# ## Find STAC Items to Load

# %%
km2deg = 1.0 / 111
x, y = (113.887, -25.843)  # Center point of a query
r = 100 * km2deg

query = catalog.search(
    collections=["sentinel-s2-l2a-cogs"],
    datetime="2021-09-16",
    limit=10,
    bbox=(x - r, y - r, x + r, y + r),
)

items = list(query.get_items())
print(f"Found: {len(items):d} datasets")

# %% [markdown]
# ## Construct Dask Dataset
#
# Note that even though there are 9 STAC Items on input, there is only one
# timeslice on output. This is because of `groupy="solar_day"`. With that
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

# %% tags=[]
from ipyleaflet import FullScreenControl, ImageOverlay, LayersControl, Map

# This compresses image with png and packs it into `data` url
# it then computes Image bounds and return `ipyleaflet.ImageOverlay`
ovr = odc.ui.mk_image_overlay(_rgba)

# Make a leaflet.Map object
lon, lat = rgba.geobox.geographic_extent.centroid.coords[0]
_map = Map(scroll_wheel_zoom=True, center=(lat, lon), zoom=8)

# Make a leaflet.Map object
_map.layout.height = "600px"
_map.add_control(FullScreenControl())
_map.add_control(LayersControl())

# Add Image overlay
_map.add_layer(ovr)

display(_map)

# %% [markdown]
# --------------------------------------------------------------
