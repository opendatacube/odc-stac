"""
Save Landsat 8 pass to GeoTIFF (COG).

This program captures one pass of Band 4 (NIR) of Lansat 8 to a single
cloud optimized GeoTIFF image. Produced image is rotated to maximize
proportion of valid pixels in the result. Data is saved in EPSG:3857 at
native resolution (30m). Produced TIFF is about 4.7GiB.

Data is sourced from Microsoft Planetary Computer:

https://planetarycomputer.microsoft.com/

Python environment

```bash
pip install odc-stac==0.3.0rc1 tqdm planetary_computer pystac-client
```

"""

import planetary_computer
import pystac_client
from affine import Affine
from dask.utils import format_bytes
from odc.geo import geom
from odc.geo.geobox import GeoBox
from tqdm.auto import tqdm

from odc.stac import configure_rio
from odc.stac import load as stac_load

res = 30  # resolution
a = 12.7  # rotation in degrees
band = "SR_B4"

catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1"
)

items = catalog.search(
    collections=["landsat-8-c2-l2"],
    datetime="2021-07-01T08:00:00Z/2021-07-01T09:00:00Z",
    bbox=(-180, -50, 180, 50),
).item_collection()

# Compute Polygon of the pass in EPSG:3857
ls8_pass = geom.unary_union(
    geom.Geometry(item.geometry, "epsg:4326").to_crs("epsg:3857") for item in items
)
assert ls8_pass is not None

# Construct rotated GeoBox
#  rotate geometry
#  construct axis aligned geobox in rotated space
#  then rotate geobox the other way
gbox = Affine.rotation(-a) * GeoBox.from_geopolygon(
    ls8_pass.transform(Affine.rotation(a)),
    resolution=res,
)

# Assume COG datasource, disables looking for external files (it's slow in the cloud)
configure_rio(cloud_defaults=True)

print(f"Loading {band} => {gbox.shape.x:,d}x{gbox.shape.y:,d}")
xx = stac_load(
    items,
    like=gbox,
    bands=[band],
    dtype="int16",
    nodata=0,
    groupby="solar_day",
    resampling="average",
    pool=4,  # Use 4 cores for loading
    progress=tqdm,  #
    patch_url=planetary_computer.sign,
)
print("Load finished")

ts = xx.time[0].dt.strftime("%Y%m%d").item()
fname = f"{band}-{ts}-{res}m.tif"
print(
    f"Will write image to: '{fname}' Raw Size is: {format_bytes(xx[band].data.size*xx[band].dtype.itemsize)}"
)

xx[band].odc.write_cog(
    fname,
    overwrite=True,
    blocksize=2048,
    ovr_blocksize=1024,
    overview_resampling="average",
    intermediate_compression={"compress": "zstd", "zstd_level": 1},
    use_windowed_writes=True,
    compress="zstd",
    zstd_level=6,
    BIGTIFF=True,
    SPARSE_OK=True,
    NUM_THREADS=4,
)
