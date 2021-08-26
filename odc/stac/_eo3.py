"""
STAC -> EO3 utilities
"""

from collections import namedtuple
from typing import Any, Dict, Iterable, List, Tuple, Iterator
from copy import deepcopy
from warnings import warn
import uuid

import toolz
from affine import Affine
import pystac.asset
import pystac.item
from pystac.extensions.projection import ProjectionExtension
from datacube.index.eo3 import prep_eo3
from datacube.index.index import default_metadata_type_docs
from datacube.model import Dataset, DatasetType, metadata_from_doc
from datacube.utils.geometry import GeoBox

BandMetadata = namedtuple("BandMetadata", ["dtype", "nodata", "units"])
ConversionConfig = Dict[str, Any]

(_eo3,) = [
    metadata_from_doc(d) for d in default_metadata_type_docs() if d.get("name") == "eo3"
]


def has_proj_ext(item: pystac.Item) -> bool:
    """
    Check if STAC Item has prjection extension
    """
    try:
        ProjectionExtension.validate_has_extension(item, add_if_missing=False)
        return True
    except pystac.ExtensionNotImplemented:
        return False


def has_proj_data(asset: pystac.asset.Asset) -> bool:
    """
    :returns: True if both ``.shape`` and ``.transform`` are set
    :returns: False if either ``.shape`` or ``.transform`` are missing
    """
    prj = ProjectionExtension.ext(asset)
    return prj.shape is not None and prj.transform is not None


def is_raster_data(asset: pystac.asset.Asset, check_proj: bool = False) -> bool:
    """
    - Has "data" role --> True
    - Has roles other than "data" --> False
    - Has no role but
      - media_type has ``image/``

    :param asset:
       STAC Asset to check

    :param check_proj:
       when enabled check if asset is part of an Item that has projection
       extension enabled and if yes only consider bands with
       projection data as "raster data" bands.
    """
    if check_proj:
        if has_proj_ext(asset.owner) and not has_proj_data(asset):
            return False

    if asset.roles is not None and len(asset.roles) > 0:
        return "data" in asset.roles
    return "image/" in asset.media_type


def asset_geobox(asset: pystac.asset.Asset) -> GeoBox:
    _proj = ProjectionExtension.ext(asset)
    assert _proj.shape is not None
    assert _proj.transform is not None
    assert _proj.crs_string is not None

    h, w = _proj.shape
    affine = Affine(*_proj.transform)
    return GeoBox(w, h, affine, _proj.crs_string)


def geobox_gsd(geobox: GeoBox) -> float:
    """
    Compute ground sampling distance of a given GeoBox.
    """
    return min(map(abs, [geobox.transform.a, geobox.transform.e]))


def compute_eo3_grids(
    assets: Dict[str, pystac.asset.Asset]
) -> Tuple[Dict[str, GeoBox], Dict[str, str]]:
    """
    Compute a minimal set of eo3 grids, pick default one, give names to
    non-default grids, while keeping track of which asset has which grid

    Assets must have ProjectionExtension with shape, transform and crs information
    populated.
    """

    def gbox_name(geobox: GeoBox) -> str:
        gsd = geobox_gsd(geobox)
        return f"g{gsd:g}"

    geoboxes = {k: asset_geobox(asset) for k, asset in assets.items()}

    # GeoBox to list of bands that share same footprint
    grids: Dict[GeoBox, List[str]] = {}

    for k, geobox in geoboxes.items():
        grids.setdefault(geobox, []).append(k)

    # Default grid is the one with highest count of bands
    #   If there is a tie pick one with the smallest ground sampling distance
    def gbox_score(geobox: GeoBox) -> Tuple[int, float]:
        return (-len(grids[geobox]), geobox_gsd(geobox))

    # locate default grid
    g_default, *_ = sorted(grids, key=gbox_score)

    named_grids: Dict[str, Geobox] = {}
    band2grid: Dict[str, str] = {}
    for grid, bands in grids.items():
        grid_name = "default" if grid is g_default else gbox_name(grid)
        if grid_name in named_grids:
            raise NotImplemented(
                "TODO: deal with multiple grids with same sampling distance"
            )

        named_grids[grid_name] = grid
        for band in bands:
            band2grid[band] = grid_name

    return named_grids, band2grid


def normalise_product_name(name: str) -> str:
    """
    Create valid product name from arbitrary string
    """

    # TODO: for now just map `-`,` ` to `_`
    return name.replace("-", "_").replace(" ", "_")


def mk_product(name: str, bands: Iterable[str], cfg: Dict[str, Any]) -> DatasetType:
    """
    Given a product name, list of bands and band metadata rules create EO3
    Datacube Product.
    """

    def _norm(meta) -> BandMetadata:
        if isinstance(meta, BandMetadata):
            return meta
        return BandMetadata(**meta)

    _cfg: Dict[str, BandMetadata] = {name: _norm(meta) for name, meta in cfg.items()}

    def make_band(name: str, cfg: Dict[str, BandMetadata]) -> Dict[str, Any]:
        info = cfg.get(name, cfg.get("*", BandMetadata("uint16", 0, "1")))

        return {
            "name": name,
            "dtype": info.dtype,
            "nodata": info.nodata,
            "units": info.units,
        }

    doc = {
        "name": normalise_product_name(name),
        "metadatat_type": "eo3",
        "measurements": [make_band(band, _cfg) for band in bands],
    }
    return DatasetType(_eo3, doc)


def infer_dc_product(item: pystac.Item, cfg: ConversionConfig) -> DatasetType:
    """
    :param item: Sample STAC Item from a collection
    :param cfg: Dictionary of configuration, see below

    .. code-block:: yaml

       sentinel-2-l2a:  # < name of the collection, i.e. ``.collection_id``
         measurements:
           "*":  # Band named "*" contains band info for "most" bands
             dtype: uint16
             nodata: 0
             units: "1"
           SCL:  # Those bands that are different than "most"
             dtype: uint8
             nodata: 0
             units: "1"
         uuid:   # Rules for constructing UUID for Datasets
             random:
             from_key: "location.of.unique.property"
             native: "location.of.key.with_actual_UUID"

       some-other-collection:
         measurements:
         #...
    """
    collection_id = item.collection_id

    cfg = deepcopy(cfg.get(collection_id, {}))

    data_bands = {
        name: asset
        for name, asset in item.assets.items()
        if is_raster_data(asset, check_proj=True)
    }

    product = mk_product(collection_id, data_bands, cfg.get("measurements", {}))

    # We assume that grouping of data bands into grids is consistent across
    # entire collection, so we compute it once and keep it on a product object
    # at least for now.
    _, band2grid = compute_eo3_grids(data_bands)
    cfg["band2grid"] = band2grid

    product._stac_cfg = cfg
    return product


def item_to_ds(item: pystac.Item, product: DatasetType) -> Dataset:
    _cfg = getattr(product, "_stac_cfg", {})

    _assets = item.assets
    data_bands = {name: _assets[name] for name in product.measurements}

    measurements: Dict[str, Dict[str, Any]] = {}
    grids: Dict[str, Dict[str, Any]] = {}
    crs = None

    for band, grid in _cfg["band2grid"].items():
        asset = _assets.get(band, None)
        if asset is None:
            warn(f"Missing asset with name: {band}")
            continue
        measurements[band] = {"path": asset.href}
        if grid != "default":
            measurements[band]["grid"] = grid

        if grid not in grids:
            geobox = asset_geobox(_assets[band])
            grids[grid] = dict(shape=geobox.shape, transform=geobox.transform)
            if crs is None:
                crs = geobox.crs
            else:
                if crs != geobox.crs:
                    raise ValueError("Expect all assets to share common CRS")

    assert crs is not None

    ds_uuid = str(uuid.uuid4())  # TODO: stop being so random

    ds_doc = {
        "id": ds_uuid,
        "$schema": "https://schemas.opendatacube.org/dataset",
        "crs": str(crs),
        "grids": grids,
        "location": "",
        "measurements": measurements,
        "properties": deepcopy(item.properties),
        "lineage": {},
    }

    return Dataset(product, prep_eo3(ds_doc), uris=[ds_doc.get("location", "")])


def stac2ds(items: Iterable[pystac.Item], cfg: ConversionConfig) -> Iterator[Dataset]:
    """
    Given a lazy sequence of STAC Items turn it into a lazy sequence of ``Dataset`` objects
    """
    products: Dict[str, DatasetType] = {}
    for item in items:
        product = products.get(item.collection_id)

        # Have not seen this collection yet, figure it out
        if product is None:
            product = infer_dc_product(item, cfg)
            products[item.collection_id] = product

        yield item_to_ds(item, product)
