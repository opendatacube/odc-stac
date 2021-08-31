"""
STAC -> EO3 utilities
"""

from collections import namedtuple
from typing import Any, Dict, Iterable, List, Set, Tuple, Iterator, Optional
from copy import deepcopy
from warnings import warn
import uuid

from affine import Affine
import pystac.asset
import pystac.item
from pystac.extensions.eo import EOExtension
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
    """
    Compute GeoBox from STAC Asset.

    This only works if ProjectionExtension is used with the
    following properties populated:

    - shape
    - transform
    - CRS

    :raises ValueError: when transform is not Affine.
    """
    _proj = ProjectionExtension.ext(asset)
    if _proj.shape is None or _proj.transform is None or _proj.crs_string is None:
        raise ValueError("The asset must have the following fields (from the projection extension): shape, transform, and one of an epsg, wkt2, or projjson")

    h, w = _proj.shape
    if len(_proj.transform) not in (6, 9):
        raise ValueError("Asset transform must be 6 or 9 elements in size")

    if len(_proj.transform) == 9 and _proj.transform[-3:] != [0, 0, 1]:
        raise ValueError(f"Asset transform is not affine: {_proj.transform}")

    affine = Affine(*_proj.transform[:6])
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

    named_grids: Dict[str, GeoBox] = {}
    band2grid: Dict[str, str] = {}
    for grid, bands in grids.items():
        grid_name = "default" if grid is g_default else gbox_name(grid)
        if grid_name in named_grids:
            raise NotImplementedError(
                "TODO: deal with multiple grids with same sampling distance"
            )

        named_grids[grid_name] = grid
        for band in bands:
            band2grid[band] = grid_name

    return named_grids, band2grid


def alias_map_from_eo(item: pystac.Item, quiet: bool = False) -> Dict[str, str]:
    """
    Generate mapping ``common name -> canonical name`` for all unique common names defined on the Item eo extension.

    :param item: STAC Item to process
    :type item: pystac.Item
    :param quiet: Do not print warning if duplicate common names are found, defaults to False
    :type quiet: bool, optional
    :return: common name to canonical name mapping
    :rtype: Dict[str, str]
    """
    try:
        bands = EOExtension.ext(item, add_if_missing=False).bands
    except pystac.ExtensionNotImplemented:
        return {}

    common_names: Dict[str, Set[str]] = {}
    for band in bands:
        common_name = band.common_name
        if common_name is not None:
            common_names.setdefault(common_name, set()).add(band.name)

    def _aliases(common_names):
        for alias, bands in common_names.items():
            if len(bands) == 1:
                (band,) = bands
                yield (alias, band)
            elif not quiet:
                warn(f"Common name `{alias}` is repeated, skipping")

    return dict(_aliases(common_names))


def normalise_product_name(name: str) -> str:
    """
    Create valid product name from arbitrary string
    """

    # TODO: for now just map `-`,` ` to `_`
    return name.replace("-", "_").replace(" ", "_")


def mk_product(
    name: str,
    bands: Iterable[str],
    cfg: Dict[str, Any],
    aliases: Optional[Dict[str, str]] = None,
) -> DatasetType:
    """
    Generate ODC Product from simplified config.

    :param name: Product name
    :type name: str
    :param bands: List of band names
    :type bands: Iterable[str]
    :param cfg: Band configuration, band_name -> Config mapping
    :type cfg: Dict[str, Any]
    :param aliases: Map of aliases ``alias -> band name``
    :type aliases: Optional[Dict[str, str]], optional
    :return: Constructed ODC Product with EO3 metadata type
    :rtype: DatasetType
    """

    if aliases is None:
        aliases = {}

    def _norm(meta) -> BandMetadata:
        if isinstance(meta, BandMetadata):
            return meta
        return BandMetadata(**meta)

    _cfg: Dict[str, BandMetadata] = {name: _norm(meta) for name, meta in cfg.items()}
    band_aliases: Dict[str, List[str]] = {}
    for alias, canonical_name in aliases.items():
        band_aliases.setdefault(canonical_name, []).append(alias)

    def make_band(
        name: str, cfg: Dict[str, BandMetadata], band_aliases: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        info = cfg.get(name, cfg.get("*", BandMetadata("uint16", 0, "1")))
        aliases = band_aliases.get(name)

        doc = {
            "name": name,
            "dtype": info.dtype,
            "nodata": info.nodata,
            "units": info.units,
        }
        if aliases is not None:
            doc["aliases"] = aliases
        return doc

    doc = {
        "name": normalise_product_name(name),
        "metadata_type": "eo3",
        "measurements": [make_band(band, _cfg, band_aliases) for band in bands],
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
         aliases:  #< unique alias -> canonical map
            rededge: B05
            rededge1: B05
            rededge2: B06
            rededge3: B07
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

    aliases = alias_map_from_eo(item)
    aliases.update(cfg.get("aliases", {}))

    product = mk_product(
        collection_id, data_bands, cfg.get("measurements", {}), aliases
    )

    # We assume that grouping of data bands into grids is consistent across
    # entire collection, so we compute it once and keep it on a product object
    # at least for now.
    _, band2grid = compute_eo3_grids(data_bands)
    cfg["band2grid"] = band2grid

    product._stac_cfg = cfg  # pylint: disable=protected-access
    return product


def item_to_ds(item: pystac.Item, product: DatasetType) -> Dataset:
    """
    Construct Dataset object from STAC Item and previosuly constructed Product.

    :raises ValueError: when not all assets share the same CRS
    """
    _cfg = getattr(product, "_stac_cfg", {})

    _assets = item.assets

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
            elif crs != geobox.crs:
                raise ValueError(
                    "Expect all assets to share common CRS"
                )  # pragma: no cover

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
