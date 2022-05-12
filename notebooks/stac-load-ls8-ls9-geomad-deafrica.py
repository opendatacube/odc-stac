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

# %% [markdown] tags=[]
# # Access the Annual Landsat 8 and 9 GeoMAD Product from Digital Earth Africa

# %% [markdown] tags=[]
# [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/opendatacube/odc-stac/develop?labpath=notebooks%2Fstac-load-ls8-ls9-geomad-deafrica.ipynb)
#
# https://explorer.digitalearth.africa/products/gm_ls8_ls9_annual

# %% [markdown]
# ## Background
#
# The Digital Earth Africa (DE Africa) GeoMAD (**Geo**median and **M**edian **A**bsolute **D**eviations) is a cloud-free composite of satellite data compiled for over annual and semi-annual (six-month) periods during each calendar year.
#
# The following GeoMAD products are available from DE Africa:
#
# * `gm_s2_annual`: Annual (calendar year) GeoMAD composite using Sentinel-2 imagery, available for the years **2017 - present**
# * `gm_s2_semiannual`: bi-annual (Jan-Jun, Jul-Dec) GeoMAD composites using Sentinel-2 imagery, available for the years **2017 - present**
# * `gm_ls8_ls9_annual`: Annual (calendar year) GeoMAD composite using Landsat-8 and Landsat-9 imagery, available for the years **2021 - present**
# * `gm_ls8_annual`: Annual (calendar year) GeoMAD composite using Landsat-8 imagery, available for the years **2013 - 2020**
# * `gm_ls5_ls7_annual`: Annual (calendar year) GeoMAD composite combining both Landsat-5 and Landsat-7 imagery, available for the years **1984 - 2012**
#
# Each product combines measurements collected over a defined period (annual or semi-annual) to produce one representative, multi-spectral image for every pixel of the African continent. 
# The end result is a comprehensive dataset that can be used either to generate true-colour images for visual inspection of the landsacpe, or the full spectral dataset can be used to develop more complex algorithms.  
#
# For a detailed description on how the GeoMAD is calculated, see the [GeoMAD technical specifications](https://docs.digitalearthafrica.org/en/latest/data_specs/GeoMAD_specs.html).
#
# **Important details:**
#
# * Datacube product names: `gm_s2_annual`, `gm_s2_semiannual`, `gm_ls8_ls9_annual`, `gm_ls8_annual`, `gm_ls5_ls7_annual`
# * Geomedian surface reflectance product
#     * Valid scaling range: `1 - 10,000`
#     * `0` is `no data`
# * Median Absolute Deviation product
#     * Valid scaling range: Spectral MAD: `0 - 1` , Bray-Curtis MAD `0 - 1`, Euclidean MAD `0 - 10,000` 
#     * `NaN` is `nodata`
# * Status: Operational
# * Date-range: 1984 &ndash; present
# * Spatial resolution: 10m for S2 products, 30m for Landsat products
#
# >Note: For a detailed description of DE Africa's GeoMAD service, see the DE Africa [GeoMAD technical specifications](https://docs.digitalearthafrica.org/en/latest/data_specs/GeoMAD_specs.html).

# %% [markdown]
# ## Description
#
# In this notebook, we will demonstrate a simple analysis workflow based on the Annual  Landsat-8 and Landsat-9 GeoMAD product. 
#
# We will load the Annual Landsat-8 and Landsat-9 GeoMAD data using the `odc` `stac_load` function then calculate the Modified Normalized Difference Water Index (MNDWI).
# We will then compare the results of the water classification of the MNDWI index to the WOfS Annual Summaries product. 

# %% [markdown]
# ## Load Packages

# %%
import pprint

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from get_product_config import get_product_config
from pystac_client import Client

from odc.stac import configure_rio, stac_load

# %% [markdown]
# ## Set Collection Configuration

# %% [markdown]
# The purpose of the configuration dictionary is to supply some optional STAC extensions that a data source might be missing. This missing information includes,  pixel data type, nodata value, unit attribute and band aliases. The configuration dictionary is passed to the `odc.stac.load` `stac_cfg=` parameter in order to supply the missing information at load time. 
#
# The configuration is per collection per asset and is determined from the product's definition. The Annual Landsat-8 and Landsat-9 GeoMAD product definition is available at https://explorer.digitalearth.africa/products/gm_ls8_ls9_annual.

# %%
product_name = "gm_ls8_ls9_annual"
# Set the profile to specify that the product is a Digital Earth Africa product.
profile = "deafrica"
config = get_product_config(product_name, profile)
pprint.pprint(config)

# %% [markdown]
# ## Set AWS Configuration

# %% [markdown]
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
# Open the stac catalogue.
catalog = Client.open("https://explorer.digitalearth.africa/stac")

# %% [markdown]
# ## Find STAC Items to Load

# %% [markdown]
# ### Define query parameters

# %% [markdown]
# >**Note**: The Annual  Landsat-8 and Landsat-9 GeoMAD composite is available for the years **2021** - **present**.

# %% [markdown]
# One way to set the study area/bounding box is to set a central latitude and longitude coordinate pair, `(central_lat, central_lon)`, then specify how many degrees to include either side of the central latitude and longitude, known as the `buffer`.
# Together, these parameters specify a square study area, as shown below:

# %% [markdown]
# <img src=data:image/png;base64,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 width="250" height="250">

# %%
# Set the central latitude and longitude.
central_lat = -5.9460
central_lon = 35.5188

# Set the buffer to load around the central coordinates.
buffer = 0.03

# Compute the bounding box for the study area
study_area_lat = (central_lat - buffer, central_lat + buffer)
study_area_lon = (central_lon - buffer, central_lon + buffer)

# Set the bounding box.
# [xmin, ymin, xmax, ymax] in latitude and longitude (EPSG:4326).
bbox = [study_area_lon[0], study_area_lat[0], study_area_lon[1], study_area_lat[1]]

# %%
# Set a start and end date.
start_date = "2021"
end_date = "2021"

# Set the STAC collections.
collections = [product_name]

# %% [markdown]
# ### Construct a query and get items from the Digital Earth Africa STAC Catalog

# %%
# Build a query with the set parameters
query = catalog.search(
    bbox=bbox, collections=collections, datetime=f"{start_date}/{end_date}"
)

# Search the STAC catalog for all items matching the query
items = list(query.get_items())
print(f"Found: {len(items):d} datasets")

# %% [markdown]
# ## Load the GeoMAD data 

# %% [markdown]
# In this step, we specify the desired coordinate system, resolution (here 30m), and bands to load.  We will load 2 spectral satellite bands: `green` and `swir_1`. Since the band aliases are contained in the `config` dictionary, bands can be loaded using these aliases instead of the band number e.g. `"swir_1"` instead of `"SR_B6"`. 
#
# We also pass the bounding box to the `stac_load` function to only load the requested data. The data will be lazy-loaded with dask, meaning that is won't be loaded into memory until necessary, such as when it is displayed.

# %%
# Specify the bands to load, the desired crs and resolution.
measurements = ("green", "swir_1")
crs = "EPSG:6933"
resolution = 30

# %%
# Load the dataset.
ds_ls = stac_load(
    items,
    bands=measurements,
    crs=crs,
    resolution=resolution,
    chunks={},
    stac_cfg=config,
    bbox=bbox,
).squeeze()

# %%
# View the xarray.Dataset.
ds_ls

# %% [markdown] tags=[]
# ## Compute the MNDWI index
#
# After loading the data, you can perform standard `xarray` operations, such as calculating the Modified Normalized Difference Water Index (MNDWI).
#
# $$
# \begin{aligned}
# \text{MNDWI} = \frac{\text{Green} - \text{SWIR}}{\text{Green} + \text{SWIR}}
# \end{aligned}
# $$
#
# >**Note:** The `.compute()` method triggers Dask to load the data into memory.

# %%
# Normalize the data by dividing the data by 10,000.
ds_ls = ds_ls / 10000
# Calculate the MNDWI index.
ds_ls["MNDWI"] = (ds_ls.green - ds_ls.swir_1) / (ds_ls.green + ds_ls.swir_1)
# Convert the xarray.Dataset to a DataArray.
mndwi = ds_ls.MNDWI.compute()

# %% [markdown]
# If a pixel's `MNDWI` value is greater than `0`, i.e. `MNDWI`>`0` then the pixel is classified as water.

# %%
water_mndwi = mndwi.where(mndwi > 0.5, np.nan)
water_mndwi = water_mndwi.where(np.isnan(water_mndwi), 1)

# %% [markdown]
# ## Load the WOfS Annual Summaries

# %%
# Set the collection configuration.
product_name = "wofs_ls_summary_annual"
config = get_product_config(product_name, profile)

# Set the STAC collections.
collections = [product_name]

# Build a query with the set parameters.
query = catalog.search(
    bbox=bbox, collections=collections, datetime=f"{start_date}/{end_date}"
)

# Search the STAC catalog for all items matching the query.
items = list(query.get_items())
print(f"Found: {len(items):d} datasets")

# %%
# Specify the bands to load.
measurements = "frequency"

# Load the dataset.
ds_wofs_annual = stac_load(
    items,
    bands=measurements,
    crs=crs,
    resolution=resolution,
    chunks={},
    stac_cfg=config,
    bbox=bbox,
).squeeze()

# View the xarray.Dataset.
ds_wofs_annual

# %%
# Convert the xarray.Dataset to a DataArray.
wofs_annual = ds_wofs_annual.frequency.compute()

# %% [markdown]
# If the frequency with which a pixel is classified as water is greater than `0.20`, i.e. `wofs_annual`  > `0.20`, then the pixel classified as regular open water during the year.

# %%
water_wofs_annual = wofs_annual.where(wofs_annual > 0.20, np.nan)
water_wofs_annual = water_wofs_annual.where(np.isnan(water_wofs_annual), 1)

# %% [markdown]
# ## Plot the MNDWI and WOfS water extents

# %%
# Plot.
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
water_mndwi.plot(ax=ax[0])
water_wofs_annual.plot(ax=ax[1])

ax[0].set_title("Landsat 8-9 GeoMAD MNDWI water extent 2021")
ax[1].set_title("WOfS Annual Summary water extent 2021")
plt.tight_layout();
