# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: ODC
#     language: python
#     name: odc
# ---

# %% [markdown]
# # Access Sentinel 2 Data from AWS
#
# https://registry.opendata.aws/sentinel-2-l2a-cogs/

# %%
import odc.ui
import yaml
from odc.algo import to_rgba
from odc.stac import stac2ds, stac_load
from pystac_client import Client

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
cfg = yaml.load(cfg, Loader=yaml.CSafeLoader)

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
# Note that even though there are 9 STAC Items on input, there is only one timeslice on output. This is because of `groupy="solar_day"`. With that setting `stac_load` will place all items that occured on the same day (as adjusted for the timezone) into one image plane.

# %%
# crs = "epsg:32749"  # native UTM projection in the query region
crs = "epsg:3857"  # Since we will plot it on a map we need to use 3857 projection
zoom = 2 ** 5  # overview level 5

xx = stac_load(
    items,
    output_crs=crs,
    resolution=(-10 * zoom, 10 * zoom),
    chunks={},
    groupby="solar_day",
    measurements=["red", "green", "blue"],
    stac_cfg=cfg,
)
xx

# %% [markdown]
# ## Convert to RGB and load

# %%
# %%time

rgba = to_rgba(xx, clamp=(1, 3000))
_rgba = rgba.compute()

# %% [markdown]
# ## Display Image on a map

# %%
dss = list(stac2ds(items, cfg))
_map = odc.ui.show_datasets(dss, style={"fillOpacity": 0.1}, scroll_wheel_zoom=True)
ovr = odc.ui.mk_image_overlay(_rgba)
_map.add_layer(ovr)
_map

# %% [markdown]
# --------------------------------------------------------------
