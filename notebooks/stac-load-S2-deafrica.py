# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: 'Python 3.8.12 64-bit (''stac'': conda)'
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
# ## Import required packages

# %%
import rasterio
from pystac_client import Client

from odc.stac import stac_load

# %% [markdown]
# ## Set configuration
#
# The configuration dictionary is determined from the product's definition, availble at https://explorer.digitalearth.africa/products/s2_l2a#definition-doc
#
# All assets except SLC have the same configuration. SLC uses `uint8` rather than `uint16`.
#
# In the configuration, we also supply the aliases for each band. This means we can load data by band name rather than band number.

# %%
config = {
    "s2_l2a": {
        "assets": {
            "*": {
                "data_type": "uint16",
                "nodata": 0, 
                "unit": "1",
            },
            "SLC": {
                "data_type": "uint8",
                "nodata": 0, 
                "unit": "1",
            },
        },
        "aliases": {
            "costal_aerosol": "B01",
            "blue": "B02",
            "green": "B03",
            "red": "B04",
            "red_edge_1": "B05",
            "red_edge_2": "B06",
            "red_edge_3": "B07",
            "nir": "B08",
            "nir_narrow": "B08A",
            "water_vapour": "B09",
            "swir_1": "B11",
            "swir_2": "B12",
            "mask": "SLC",
            "aerosol_optical_thickness": "AOT",
            "scene_average_water_vapour": "WVP",
        }
    }
}

# %% [markdown]
# ## Connect to the Digital Earth Africa STAC catalog

# %%
# Open the stac catalogue
deafrica_stac_address = 'https://explorer.digitalearth.africa/stac'
catalog = Client.open(deafrica_stac_address)

# %% [markdown]
# ## Find STAC Items to Load

# %%
# Construct a bounding box to search over
# [xmin, ymin, xmax, ymax] in latitude and longitude
bbox = [37.76, 12.49, 37.77, 12.50]

# Construct a time range to search over
start_date = '2020-09-01'
end_date = '2020-12-01'
timerange = f'{start_date}/{end_date}'

# Choose the product/s to load
products = ['s2_l2a']

# %%
# Identify all data matching the above:
query = catalog.search(
    bbox=bbox,
    collections=products,
    datetime=timerange
)

items = list(query.get_items())
print(f"Found: {len(items):d} datasets")

# %% [markdown]
# ## Construct Dask Dataset
#
# In this step, we specify the desired coordinate system, resolution (here 20m), and bands to load. We also pass the bounding box to the `stac_load` function to only load the requested data.

# %%
crs = 'EPSG:6933'
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


# %%
# View the Xarray Dataset
ds

# %% [markdown]
# ## Load the data into memory
#
# Digital Earth Africa data is stored on S3 in Cape Town, Africa. To load the data, we must use a rasterio Env configured with the appropriate AWS S3 endpoint.

# %%
# Load into memory
with rasterio.Env(AWS_S3_ENDPOINT='s3.af-south-1.amazonaws.com', AWS_NO_SIGN_REQUEST='YES'):
    ds_loaded = ds.load()

# %% [markdown]
# ### Compute a band index
#
# After loading the data, you can perform standard Xarray operations, such as calculating and plotting the normalised difference vegetation index (NDVI).

# %%
ds_loaded["NDVI"] = (ds_loaded.nir - ds_loaded.red)/(ds_loaded.nir + ds_loaded.red)

ds_loaded.NDVI.plot(col="time", col_wrap=6, vmin=0, vmax=1);
