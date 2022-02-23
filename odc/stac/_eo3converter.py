"""
STAC -> EO3 utilities.

Utilities for translating STAC Items to EO3 Datasets.
"""

import uuid
from copy import copy
from functools import partial, singledispatch
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence
from warnings import warn

import pystac.asset
import pystac.collection
import pystac.errors
import pystac.item
from datacube.index.eo3 import prep_eo3
from datacube.index.index import default_metadata_type_docs
from datacube.model import Dataset, DatasetType, metadata_from_doc
from odc.geo import Geometry
from toolz import dicttoolz

from ._mdtools import (
    EPSG4326,
    ConversionConfig,
    RasterBandMetadata,
    alias_map_from_eo,
    asset_geobox,
    band2grid_from_gsd,
    band_metadata,
    compute_eo3_grids,
    has_proj_ext,
    is_raster_data,
    mk_1x1_geobox,
    mk_sample_item,
    norm_band_metadata,
    normalise_product_name,
)

# uuid.uuid5(uuid.NAMESPACE_URL, "https://stacspec.org")
UUID_NAMESPACE_STAC = uuid.UUID("55d26088-a6d0-5c77-bf9a-3a7f3c6a6dab")

# Mapping between EO3 field names and STAC properties object field names
# EO3 metadata was defined before STAC 1.0, so we used some extensions
# that are now part of the standard instead
STAC_TO_EO3_RENAMES = {
    "end_datetime": "dtr:end_datetime",
    "start_datetime": "dtr:start_datetime",
    "gsd": "eo:gsd",
    "instruments": "eo:instrument",
    "platform": "eo:platform",
    "constellation": "eo:constellation",
    "view:off_nadir": "eo:off_nadir",
    "view:azimuth": "eo:azimuth",
    "view:sun_azimuth": "eo:sun_azimuth",
    "view:sun_elevation": "eo:sun_elevation",
}

(_eo3,) = (
    metadata_from_doc(d) for d in default_metadata_type_docs() if d.get("name") == "eo3"
)


def mk_product(
    name: str,
    bands: Iterable[str],
    cfg: Dict[str, Any],
    aliases: Optional[Dict[str, str]] = None,
) -> DatasetType:
    """
    Generate ODC Product from simplified config.

    :param name: Product name
    :param bands: List of band names
    :param cfg: Band configuration, band_name -> Config mapping
    :param aliases: Map of aliases ``alias -> band name``
    :return: Constructed ODC Product with EO3 metadata type
    """
    if aliases is None:
        aliases = {}

    _cfg: Dict[str, RasterBandMetadata] = {
        name: norm_band_metadata(meta) for name, meta in cfg.items()
    }
    band_aliases: Dict[str, List[str]] = {}
    for alias, canonical_name in aliases.items():
        band_aliases.setdefault(canonical_name, []).append(alias)

    def make_band(
        name: str,
        cfg: Dict[str, RasterBandMetadata],
        band_aliases: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        info = cfg.get(name, cfg.get("*", RasterBandMetadata("uint16", 0, "1")))
        aliases = band_aliases.get(name)

        # map to ODC names for raster:bands
        doc = {
            "name": name,
            "dtype": info.data_type,
            "nodata": info.nodata,
            "units": info.unit,
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


def _collection_id(item: pystac.item.Item) -> str:
    if item.collection_id is None:
        # workaround for some early ODC data
        return str(item.properties.get("odc:product", "_"))
    return str(item.collection_id)


@singledispatch
def infer_dc_product(x: Any, cfg: Optional[ConversionConfig] = None) -> DatasetType:
    """Overloaded function."""
    raise TypeError(
        "Invalid type, must be one of: pystac.item.Item, pystac.collection.Collection"
    )


@infer_dc_product.register(pystac.item.Item)
def infer_dc_product_from_item(
    item: pystac.item.Item, cfg: Optional[ConversionConfig] = None
) -> DatasetType:
    """
    Infer Datacube product object from a STAC Item.

    :param item: Sample STAC Item from a collection
    :param cfg: Dictionary of configuration, see below

    .. code-block:: yaml

       sentinel-2-l2a:  # < name of the collection, i.e. ``.collection_id``
         assets:
           "*":  # Band named "*" contains band info for "most" bands
             data_type: uint16
             nodata: 0
             unit: "1"
           SCL:  # Those bands that are different than "most"
             data_type: uint8
             nodata: 0
             unit: "1"
         aliases:  #< unique alias -> canonical map
           rededge: B05
           rededge1: B05
           rededge2: B06
           rededge3: B07
         uuid:          # Rules for constructing UUID for Datasets
           mode: auto   # auto|random|native(expect .id to contain valid UUID string)
           extras:      # List of extra keys from properties to include (mode=auto)
           - "s2:generation_time"

         warnings: ignore  # ignore|all  (default all)

       some-other-collection:
         assets:
         #...

       "*": # Applies to all collections if not defined on a collection
         warnings: ignore
    """
    # pylint: disable=too-many-locals
    if cfg is None:
        cfg = {}

    collection_id = _collection_id(item)

    _cfg = copy(cfg.get("*", {}))
    _cfg.update(cfg.get(collection_id, {}))

    quiet = _cfg.get("warnings", "all") == "ignore"
    band_cfg = _cfg.get("assets", {})
    ignore_proj = _cfg.get("ignore_proj", False)

    def _keep(kv, check_proj):
        name, asset = kv
        if name in band_cfg:
            return True
        return is_raster_data(asset, check_proj=check_proj)

    has_proj = False if ignore_proj else has_proj_ext(item)
    data_bands: Dict[str, pystac.asset.Asset] = dicttoolz.itemfilter(
        partial(_keep, check_proj=has_proj), item.assets
    )
    if len(data_bands) == 0 and has_proj is True:
        # Proj is enabled but no Asset has all the proj data
        has_proj = False
        data_bands = dicttoolz.itemfilter(partial(_keep, check_proj=False), item.assets)
        _cfg.update(ignore_proj=True)

    if len(data_bands) == 0:
        raise ValueError("Unable to find any bands")

    aliases = alias_map_from_eo(item, quiet=quiet)
    aliases.update(_cfg.get("aliases", {}))

    # 1. If band in user config -- use that
    # 2. Use data from raster extension (with fallback to "*" config)
    # 3. Use config for "*" from user config as fallback
    band_defaults = norm_band_metadata(band_cfg.get("*", {}))
    for name, asset in data_bands.items():
        if name not in band_cfg:
            bm = band_metadata(asset, band_defaults)
            if bm is not band_defaults:
                band_cfg[name] = bm

    product = mk_product(collection_id, data_bands, band_cfg, aliases)

    # We assume that grouping of data bands into grids is consistent across
    # entire collection, so we compute it once and keep it on a product object
    # at least for now.
    if has_proj:
        _, band2grid = compute_eo3_grids(data_bands)
    else:
        band2grid = band2grid_from_gsd(data_bands)

    _cfg["band2grid"] = band2grid
    setattr(product, "_stac_cfg", _cfg)  # pylint: disable=protected-access
    return product


def _compute_uuid(
    item: pystac.item.Item, mode: str = "auto", extras: Optional[Sequence[str]] = None
) -> uuid.UUID:
    if mode == "native":
        return uuid.UUID(item.id)
    if mode == "random":
        return uuid.uuid4()

    assert mode == "auto"
    # 1. see if .id is already a UUID
    try:
        return uuid.UUID(item.id)
    except ValueError:
        pass

    # 2. .collection_id, .id, [extras]
    #
    # Deterministic UUID is using uuid5 on a string constructed from Item properties like so
    #
    #  <collection_id>\n
    #  <item_id>\n
    #  extras[i]=item.properties[extras[i]]\n
    #
    #  At a minimum it's just 2 lines collection_id and item.id If extra keys are requested, these
    #  are sorted first and then appended one per line in `{key}={value}` format where value is
    #  looked up from item properties, if key is missing then {value} is set to empty string.
    hash_srcs = [_collection_id(item), item.id]
    if extras is not None:
        tags = [f"{k}={str(item.properties.get(k, ''))}" for k in sorted(extras)]
        hash_srcs.extend(tags)
    hash_text = "\n".join(hash_srcs) + "\n"  # < ensure last line ends on \n
    return uuid.uuid5(UUID_NAMESPACE_STAC, hash_text)


def item_to_ds(item: pystac.item.Item, product: DatasetType) -> Dataset:
    """
    Construct Dataset object from STAC Item and previously constructed Product.

    :raises ValueError: when not all assets share the same CRS
    """
    # pylint: disable=too-many-locals
    _cfg = getattr(product, "_stac_cfg", {})
    band2grid: Dict[str, str] = _cfg.get("band2grid", {})
    ignore_proj: bool = _cfg.get("ignore_proj", False)

    has_proj = False if ignore_proj else has_proj_ext(item)
    measurements: Dict[str, Dict[str, Any]] = {}
    grids: Dict[str, Dict[str, Any]] = {}
    crs = None
    _assets = item.assets

    for band in product.measurements:
        asset = _assets.get(band, None)
        if asset is None:
            warn(f"Missing asset with name: {band}")
            continue
        measurements[band] = {"path": asset.href}

        # Only compute grids when proj extension is enabled
        if not has_proj:
            continue

        grid_name = band2grid.get(band, "default")
        if grid_name != "default":
            measurements[band]["grid"] = grid_name

        if grid_name not in grids:
            geobox = asset_geobox(_assets[band])
            grids[grid_name] = dict(shape=geobox.shape, transform=geobox.transform)
            if crs is None:
                crs = geobox.crs
            elif crs != geobox.crs:
                raise ValueError(
                    "Expect all assets to share common CRS"
                )  # pragma: no cover

    # No proj metadata: make up 1x1 Grid in EPSG4326 instead
    if not has_proj:
        # TODO: support partial proj, when only CRS is known but not shape/transform
        # - get native CRS
        # - compute bounding box in native CRS
        # - construct 1x1 geobox in native CRS

        geom = Geometry(item.geometry, EPSG4326)
        geobox = mk_1x1_geobox(geom)
        crs = geobox.crs
        grids["default"] = dict(shape=geobox.shape, transform=geobox.transform)

    assert crs is not None

    uuid_cfg = _cfg.get("uuid", {})
    ds_uuid = _compute_uuid(
        item, mode=uuid_cfg.get("mode", "auto"), extras=uuid_cfg.get("extras", [])
    )

    ds_doc = {
        "id": str(ds_uuid),
        "$schema": "https://schemas.opendatacube.org/dataset",
        "crs": str(crs),
        "grids": grids,
        "location": "",
        "measurements": measurements,
        "properties": dicttoolz.keymap(
            lambda k: STAC_TO_EO3_RENAMES.get(k, k), item.properties
        ),
        "lineage": {},
    }

    return Dataset(product, prep_eo3(ds_doc), uris=[ds_doc.get("location", "")])


def stac2ds(
    items: Iterable[pystac.item.Item],
    cfg: Optional[ConversionConfig] = None,
    product_cache: Optional[Dict[str, DatasetType]] = None,
) -> Iterator[Dataset]:
    """
    STAC :class:`~pystac.item.Item` to :class:`~datacube.model.Dataset` stream converter.

    Given a lazy sequence of STAC :class:`~pystac.item.Item` objects turn it into a lazy sequence of
    :class:`~datacube.model.Dataset` objects.

    .. rubric:: Assumptions

    First observed :py:class:`~pystac.item.Item` for a given collection is used to construct
    :py:mod:`datacube` product definition. After that, all subsequent items from the same collection
    are interpreted according to that product spec. Specifically this means that every item is
    expected to have the same set of bands. If product contains bands with different resolutions, it
    is assumed that the same set of bands share common resolution across all items in the
    collection.

    :param items:
       Lazy sequence of :class:`~pystac.item.Item` objects

    :param cfg:
       Supply metadata missing from STAC, configure aliases, control warnings

    :param product_cache:
       Input/Output parameter, contains mapping from collection name to deduced product definition,
       i.e. :py:class:`datacube.model.DatasetType` object.

    .. rubric: Sample Configuration

    .. code-block:: yaml

       sentinel-2-l2a:  # < name of the collection, i.e. `.collection_id`
         assets:
           "*":  # Band named "*" contains band info for "most" bands
             data_type: uint16
             nodata: 0
             unit: "1"
           SCL:  # Those bands that are different than "most"
             data_type: uint8
             nodata: 0
             unit: "1"
         aliases:  #< unique alias -> canonical map
           rededge: B05
           rededge1: B05
           rededge2: B06
           rededge3: B07
         uuid:          # Rules for constructing UUID for Datasets
           mode: auto   # auto|random|native(expect .id to contain valid UUID string)
           extras:      # List of extra keys from properties to include (mode=auto)
           - "s2:generation_time"

         warnings: ignore  # ignore|all  (default all)

       some-other-collection:
         assets:
         #...

       "*": # Applies to all collections if not defined on a collection
         warnings: ignore

    """
    products: Dict[str, DatasetType] = {} if product_cache is None else product_cache
    for item in items:
        collection_id = item.collection_id or "_"
        collection_id = str(collection_id)
        product = products.get(collection_id)

        # Have not seen this collection yet, figure it out
        if product is None:
            product = infer_dc_product(item, cfg)
            products[collection_id] = product

        yield item_to_ds(item, product)


@infer_dc_product.register(pystac.collection.Collection)
def infer_dc_product_from_collection(
    collection: pystac.collection.Collection, cfg: Optional[ConversionConfig] = None
) -> DatasetType:
    """
    Construct Datacube Product definition from STAC Collection.

    :param collection: STAC Collection
    :param cfg: Configuration dictionary
    """
    if cfg is None:
        cfg = {}
    return infer_dc_product(mk_sample_item(collection), cfg)
