"""
STAC -> EO3 utilities.

Utilities for translating STAC Items to EO3 Datasets.
"""

import dataclasses
import uuid
from functools import singledispatch
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

import pystac.asset
import pystac.collection
import pystac.errors
import pystac.item
from datacube.index.eo3 import prep_eo3

try:
    from datacube.index.index import default_metadata_type_docs  # type: ignore
except ImportError:
    from datacube.index.abstract import default_metadata_type_docs  # type: ignore

from datacube.model import Dataset, DatasetType, metadata_from_doc
from odc.geo import CRS
from odc.geo.geobox import GeoBox
from toolz import dicttoolz

from .._mdtools import (
    EPSG4326,
    ConversionConfig,
    _collection_id,
    extract_collection_metadata,
    mk_1x1_geobox,
    mk_sample_item,
    parse_item,
)
from .._model import BandKey, ParsedItem, RasterBandMetadata, RasterCollectionMetadata

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


def _to_product(md: RasterCollectionMetadata) -> DatasetType:
    def make_band(
        band_key: BandKey,
        band: RasterBandMetadata,
        band_aliases: Dict[BandKey, List[str]],
    ) -> Dict[str, Any]:
        name, idx = band_key
        if idx > 1:
            name = f"{name}_{idx}"
        aliases = band_aliases.get(band_key)

        # map to ODC names for raster:bands
        doc: Dict[str, Any] = {
            "name": name,
            "dtype": band.data_type,
            "nodata": band.nodata,
            "units": band.unit,
        }
        if aliases is not None:
            doc["aliases"] = aliases
        if idx > 0:
            doc["band"] = idx + 1  # one based in datacube
        return doc

    # drop ambigous aliases
    band_aliases = md.band_aliases(unique=True)
    doc = {
        "name": md.name,
        "metadata_type": "eo3",
        "metadata": {"product": {"name": md.name}},
        "measurements": [
            make_band(band_key, band, band_aliases)
            for band_key, band in md.bands.items()
        ],
    }
    return DatasetType(_eo3, doc)


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
    """
    md = extract_collection_metadata(item, cfg)
    product = _to_product(md)
    setattr(product, "_md", md)  # pylint: disable=protected-access
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


def _to_grid(gbox: GeoBox) -> Dict[str, Any]:
    return {"shape": gbox.shape.yx, "transform": gbox.transform[:6]}


def _to_dataset(
    item: ParsedItem,
    properties: Dict[str, Any],
    ds_uuid: uuid.UUID,
    product: DatasetType,
) -> Dataset:
    # pylint: disable=too-many-locals

    md = item.collection
    band2grid = md.band2grid
    grids: Dict[str, Dict[str, Any]] = {}
    measurements: Dict[str, Dict[str, Any]] = {}
    crs: Optional[CRS] = None

    for band_key, src in item.bands.items():
        name, idx = band_key

        m: Dict[str, Any] = {"path": src.uri}
        if idx > 1:
            m["band"] = idx
        measurements[name] = m

        if not md.has_proj:
            continue

        grid_name = band2grid.get(name, "default")
        if grid_name != "default":
            m["grid"] = grid_name

        gbox = src.geobox
        if gbox is None:
            continue

        if crs is None:
            crs = gbox.crs

        if grid_name not in grids:
            grids[grid_name] = _to_grid(gbox)

    if len(grids) == 0:
        if item.geometry is None:
            raise ValueError("Item without footprint")

        gbox = mk_1x1_geobox(item.geometry)
        grids["default"] = _to_grid(gbox)
        crs = gbox.crs

    if crs is None:
        crs = EPSG4326

    ds_doc = {
        "id": str(ds_uuid),
        "$schema": "https://schemas.opendatacube.org/dataset",
        "crs": str(crs),
        "grids": grids,
        "location": "",
        "measurements": measurements,
        "properties": dicttoolz.keymap(
            lambda k: STAC_TO_EO3_RENAMES.get(k, k), properties
        ),
        "lineage": {},
    }

    return Dataset(product, prep_eo3(ds_doc), uris=[ds_doc.get("location", "")])


def _item_to_ds(
    item: pystac.item.Item, product: DatasetType, cfg: Optional[ConversionConfig] = None
) -> Dataset:
    """
    Construct Dataset object from STAC Item and previously constructed Product.

    :raises ValueError: when not all assets share the same CRS
    """
    # pylint: disable=too-many-locals
    if cfg is None:
        cfg = {}

    md: RasterCollectionMetadata = getattr(product, "_md")
    uuid_cfg = cfg.get("uuid", {})
    ds_uuid = _compute_uuid(
        item, mode=uuid_cfg.get("mode", "auto"), extras=uuid_cfg.get("extras", [])
    )
    _item = parse_item(item, md)

    return _to_dataset(_item, item.properties, ds_uuid, product)


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
        collection_id = _collection_id(item)
        product = products.get(collection_id)

        # Have not seen this collection yet, figure it out
        if product is None:
            product = infer_dc_product(item, cfg)
            products[collection_id] = product

        yield _item_to_ds(item, product, cfg)


@infer_dc_product.register(pystac.collection.Collection)
def infer_dc_product_from_collection(
    collection: pystac.collection.Collection, cfg: Optional[ConversionConfig] = None
) -> DatasetType:
    """
    Construct Datacube Product definition from STAC Collection.

    :param collection: STAC Collection
    :param cfg: Configuration dictionary
    """
    # pylint: disable=protected-access
    if cfg is None:
        cfg = {}
    product = infer_dc_product(mk_sample_item(collection), cfg)

    # unless configured to ignore projection info assume that it will be present
    ignore_proj = cfg.get(product.name, {}).get("ignore_proj", False)
    if not ignore_proj:
        product._md = dataclasses.replace(product._md, has_proj=True)
    return product
