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
# # Access Sentinel 2 Data on Planetary Computer
#
# [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/opendatacube/odc-stac/develop?labpath=notebooks%2Fstac-load-S2-ms.ipynb)

# %% [markdown]
# ## Setup Instructions
#
# This notebook is meant to run on Planetary Computer lab hub.

# %%
import dask.distributed
import dask.utils
import numpy as np
import planetary_computer as pc
import xarray as xr
from IPython.display import display
from pystac_client import Client

from odc.stac import configure_rio, stac_load

# %% [markdown]
# ## Start Dask Client
#
# This step is optional, but it does improve load speed significantly. You
# don't have to use Dask, as you can load data directly into memory of the
# notebook.

# %%
client = dask.distributed.Client()
configure_rio(cloud_defaults=True, client=client)
display(client)

# %% [markdown]
# ## Query STAC API
#
# Here we are looking for datasets in `sentinel-2-l2a` collection from June
# 2019 over MGRS tile `06VVN`.

# %%
catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

query = catalog.search(
    collections=["sentinel-2-l2a"],
    datetime="2019-06",
    query={"s2:mgrs_tile": dict(eq="06VVN")},
)

items = list(query.items())
print(f"Found: {len(items):d} datasets")

# %% [markdown]
# ## Lazy load all the bands
#
# We won't use all the bands but it doesn't matter as bands that we won't use
# won't be loaded. We are "loading" data with Dask, which means that at this
# point no reads will be happening just yet.
#
# We have to supply `dtype=` and `nodata=` because items in this collection are missing [raster extension](https://github.com/stac-extensions/raster) metadata.

# %%
resolution = 10
SHRINK = 4
if client.cluster.workers[0].memory_manager.memory_limit < dask.utils.parse_bytes("4G"):
    SHRINK = 8  # running on Binder with 2Gb RAM

if SHRINK > 1:
    resolution = resolution * SHRINK

xx = stac_load(
    items,
    chunks={"x": 2048, "y": 2048},
    patch_url=pc.sign,
    resolution=resolution,
    # force dtype and nodata
    dtype="uint16",
    nodata=0,
)

print(f"Bands: {','.join(list(xx.data_vars))}")
display(xx)

# %% [markdown]
# By default `stac_load` will return all the data bands using canonical asset
# names. But we can also request a subset of bands, by supplying `bands=` parameter.
# When going this route you can also use "common name" to refer to a band.
#
# In this case we request `red,green,blue,nir` bands which are common names for
# bands `B04,B03,B02,B08` and `SCL` band which is a canonical name.

# %%
xx = stac_load(
    items,
    bands=["red", "green", "blue", "nir", "SCL"],
    resolution=resolution,
    chunks={"x": 2048, "y": 2048},
    patch_url=pc.sign,
    # force dtype and nodata
    dtype="uint16",
    nodata=0,
)

print(f"Bands: {','.join(list(xx.data_vars))}")
display(xx)


# %% [markdown]
# ## Do some math with bands


# %%
def to_float(xx):
    _xx = xx.astype("float32")
    nodata = _xx.attrs.pop("nodata", None)
    if nodata is None:
        return _xx
    return _xx.where(xx != nodata)


def colorize(xx, colormap):
    return xr.DataArray(colormap[xx.data], coords=xx.coords, dims=(*xx.dims, "band"))


# %%
# like .astype(float32) but taking care of nodata->NaN mapping
nir = to_float(xx.nir)
red = to_float(xx.red)
ndvi = (nir - red) / (
    nir + red
)  # < This is still a lazy Dask computation (no data loaded yet)

# Get the 5-th time slice `load->compute->plot`
_ = ndvi.isel(time=4).compute().plot.imshow(size=7, aspect=1.2, interpolation="bicubic")

# %% [markdown]
# For sample purposes work with first 6 observations only

# %%
xx = xx.isel(time=np.s_[:6])

# %%
# fmt: off
scl_colormap = np.array(
    [
        [255,   0, 255, 255],  # 0  - NODATA
        [255,   0,   4, 255],  # 1  - Saturated or Defective
        [0  ,   0,   0, 255],  # 2  - Dark Areas
        [97 ,  97,  97, 255],  # 3  - Cloud Shadow
        [3  , 139,  80, 255],  # 4  - Vegetation
        [192, 132,  12, 255],  # 5  - Bare Ground
        [21 , 103, 141, 255],  # 6  - Water
        [117,   0,  27, 255],  # 7  - Unclassified
        [208, 208, 208, 255],  # 8  - Cloud
        [244, 244, 244, 255],  # 9  - Definitely Cloud
        [195, 231, 240, 255],  # 10 - Thin Cloud
        [222, 157, 204, 255],  # 11 - Snow or Ice
    ],
    dtype="uint8",
)
# fmt: on

# Load SCL band, then convert to RGB using color scheme above
scl_rgba = colorize(xx.SCL.compute(), scl_colormap)

# Check we still have geo-registration
scl_rgba.odc.geobox

# %%
_ = scl_rgba.plot.imshow(col="time", col_wrap=3, size=3, interpolation="antialiased")

# %% [markdown]
# Let's save image dated 2019-06-04 to a cloud optimized geotiff file.

# %%
to_save = scl_rgba.isel(time=3)
fname = f"SCL-{to_save.time.dt.strftime('%Y%m%d').item()}.tif"
print(f"Saving to: '{fname}'")

# %%
scl_rgba.isel(time=3).odc.write_cog(
    fname,
    overwrite=True,
    compress="webp",
    webp_quality=90,
)

# %% [markdown]
# Check the file with `rio info`.

# %%
# !ls -lh {fname}
# !rio info {fname} | jq .

# %% [markdown]
# --------------------------------
