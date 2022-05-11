# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Access Sentinel 2 Analysis Ready Data from Digital Earth Africa
#
# [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/opendatacube/odc-stac/develop?labpath=notebooks%2Fstac-load-S2-deafrica.ipynb)
#
# https://explorer.digitalearth.africa/products/s2_l2a

# %% [markdown]
# ## Import Required Packages

# %%
import pprint

from pystac_client import Client
from odc.stac import stac_load, configure_rio
from get_product_config import get_product_config

# %% [markdown]
# ## Set Collection Configuration
#
# The purpose of the configuration dictionary is to supply some optional STAC extensions that a data source might be missing. This missing information includes,  pixel data type, nodata value, unit attribute and band aliases. The configuration dictionary is passed to the `odc.stac.load` `stac_cfg=` parameter in order to supply the missing information at load time. 
#
# The configuration is per collection per asset and is determined from the product's definition. The Sentinel-2 product definition is available at https://explorer.digitalearth.africa/products/s2_l2a.

# %%
product_name = "s2_l2a"
# Set the profile to specify that the product is a Digital Earth Africa product.
profile = "deafrica"
config = get_product_config(product_name, profile)
pprint.pprint(config)

# %% [markdown]
# ## Set AWS Configuration
#
# Digital Earth Africa data is stored on S3 in Cape Town, Africa. To load the data, we must configure rasterio with the appropriate AWS S3 endpoint. This can be done with the `odc.stac.configure_rio` function. Documentation for this function is available at https://odc-stac.readthedocs.io/en/latest/_api/odc.stac.configure_rio.html#odc.stac.configure_rio.
#
# The configuration below must be used when loading any Digital Earth Africa data through the STAC API.

# %%
configure_rio(
    cloud_defaults=True,
    aws={"aws_unsigned": True},
    AWS_S3_ENDPOINT="s3.af-south-1.amazonaws.com",
)


# %% [markdown]
# ## Connect to the Digital Earth Africa STAC Catalog

# %%
# Open the stac catalogue
catalog = Client.open("https://explorer.digitalearth.africa/stac")


# %% [markdown]
# ## Find STAC Items to Load
#
# ### Define query parameters

# %%
# Set a bounding box
# [xmin, ymin, xmax, ymax] in latitude and longitude
bbox = [37.76, 12.49, 37.77, 12.50]

# Set a start and end date
start_date = "2020-09-01"
end_date = "2020-12-01"

# Set the STAC collections
collections = [product_name]


# %% [markdown]
# ### Construct query and get items from catalog

# %%
# Build a query with the set parameters
query = catalog.search(
    bbox=bbox, collections=collections, datetime=f"{start_date}/{end_date}"
)

# Search the STAC catalog for all items matching the query
items = list(query.get_items())
print(f"Found: {len(items):d} datasets")

# %% [markdown]
# ## Load the Data
#
# In this step, we specify the desired coordinate system, resolution (here 20m), and bands to load. We also pass the bounding box to the `stac_load` function to only load the requested data. Since the band aliases are contained in the `config` dictionary, bands can be loaded using these aliaes (e.g. `"red"` instead of `"B04"` below).
#
# The data will be lazy-loaded with dask, meaning that is won't be loaded into memory until necessary, such as when it is displayed.

# %%
crs = "EPSG:6933"
resolution = 20

ds = stac_load(
    items,
    bands=("red", "green", "blue", "nir"),
    crs=crs,
    resolution=resolution,
    chunks={},
    groupby="solar_day",
    stac_cfg=config,
    bbox=bbox,
)

# View the Xarray Dataset
ds


# %% [markdown]
# ### Compute a band index
#
# After loading the data, you can perform standard Xarray operations, such as calculating and plotting the normalised difference vegetation index (NDVI). The `.compute()` method triggers Dask to load the data into memory, so running this step may take a few minutes.

# %%
ds["NDVI"] = (ds.nir - ds.red) / (ds.nir + ds.red)


ds.NDVI.compute().plot(col="time", col_wrap=6, vmin=0, vmax=1)
