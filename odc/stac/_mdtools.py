"""
STAC -> EO3 utilities.

Utilities for translating STAC Items to EO3 Datasets.
"""

import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, TypeVar, Union
from warnings import warn

import pystac.asset
import pystac.collection
import pystac.errors
import pystac.item
from affine import Affine
from odc.geo import CRS, Geometry, wh_
from odc.geo.geobox import GeoBox
from pystac.extensions.eo import EOExtension
from pystac.extensions.item_assets import ItemAssetsExtension
from pystac.extensions.projection import ProjectionExtension
from pystac.extensions.raster import RasterExtension
from toolz import dicttoolz

from ._model import RasterBandMetadata

T = TypeVar("T")
ConversionConfig = Dict[str, Any]

EPSG4326 = CRS("EPSG:4326")

# Assets with these roles are ignored unless manually requested
ROLES_THUMBNAIL = {"thumbnail", "overview"}

# Used to detect image assets when media_type is missing
RASTER_FILE_EXTENSIONS = {"tif", "tiff", "jpeg", "jpg", "jp2", "img"}


def with_default(v: Optional[T], default_value: T) -> T:
    """
    Replace ``None`` with default value.

    :param v: Value that might be None
    :param default_value: Default value of the same type as v
    :return: ``v`` unless it is ``None`` then return ``default_value`` instead
    """
    if v is None:
        return default_value
    return v


def band_metadata(
    asset: pystac.asset.Asset, default: RasterBandMetadata
) -> RasterBandMetadata:
    """
    Compute band metadata from Asset raster extension with defaults from default.

    :param asset: Asset with raster extension
    :param default: Values to use for fallback
    :return: BandMetadata tuple constructed from raster:bands metadata
    """
    try:
        rext = RasterExtension.ext(asset)
    except pystac.errors.ExtensionNotImplemented:
        return default

    if rext.bands is None or len(rext.bands) == 0:
        return default

    if len(rext.bands) > 1:
        warn(f"Defaulting to first band of {len(rext.bands)}")
    band = rext.bands[0]

    return RasterBandMetadata(
        with_default(band.data_type, default.data_type),
        with_default(band.nodata, default.nodata),
        with_default(band.unit, default.unit),
    )


def has_proj_ext(item: Union[pystac.item.Item, pystac.collection.Collection]) -> bool:
    """
    Check if STAC Item or Collection has projection extension.

    :returns: ``True`` if PROJ exetension is enabled
    :returns: ``False`` if no PROJ extension was found
    """
    try:
        ProjectionExtension.validate_has_extension(item, add_if_missing=False)
        return True
    except pystac.errors.ExtensionNotImplemented:
        return False


def has_proj_data(asset: pystac.asset.Asset) -> bool:
    """
    Check if STAC Asset contains proj extension data.

    :returns: True if both ``.shape`` and ``.transform`` are set
    :returns: False if either ``.shape`` or ``.transform`` are missing
    """
    prj = ProjectionExtension.ext(asset)
    return prj.shape is not None and prj.transform is not None


def is_raster_data(asset: pystac.asset.Asset, check_proj: bool = False) -> bool:
    """
    Heuristic for determining if Asset points to raster data.

    - If media type looks like image and roles don't look like thumbnail/overview
    - If media type is undefined and roles contains "data"
    - If media type is undefined and href ends on image extension

    :param asset:
       STAC Asset to check

    :param check_proj:
       when enabled check if asset is part of an Item that has projection
       extension enabled and if yes only consider bands with
       projection data as "raster data" bands.
    """
    # pylint: disable=too-many-return-statements
    #   some of these are for readability

    if check_proj:
        if (
            asset.owner is not None
            and has_proj_ext(asset.owner)
            and not has_proj_data(asset)
        ):
            return False

    roles: Set[str] = set(asset.roles or [])

    media_type = asset.media_type
    if media_type is None:
        # Type undefined
        #   Look if it has data role
        if "data" in roles:
            return True
        if "metadata" in roles:
            return False
    elif "image/" in media_type:
        # Image:
        #    False -- when thumbnail
        #    True  -- otherwise
        if any(r in roles for r in ROLES_THUMBNAIL):
            return False
        return True
    else:
        # Some type that is not `image/*`
        return False

    ext = asset.href.split(".")[-1].lower()
    return ext in RASTER_FILE_EXTENSIONS


def mk_1x1_geobox(geom: Geometry) -> GeoBox:
    """
    Construct 1x1 pixels GeoBox tightly enclosing supplied geometry.

    :param geom: Geometry in whatever projection
    :return: GeoBox object such that geobox.extent.contains(geom) is True, geobox.shape == (1,1)
    """
    x1, y1, x2, y2 = geom.boundingbox
    # note that Y axis is inverted
    #   0,0 -> X_min, Y_max
    #   1,1 -> X_max, Y_min
    return GeoBox((1, 1), Affine((x2 - x1), 0, x1, 0, (y1 - y2), y2), geom.crs)


def asset_geobox(asset: pystac.asset.Asset) -> GeoBox:
    """
    Compute GeoBox from STAC Asset.

    This only works if ProjectionExtension is used with the
    following properties populated:

    - shape
    - transform
    - CRS

    :raises ValueError: when transform,shape or crs are missing
    :raises ValueError: when transform is not Affine.
    """
    try:
        _proj = ProjectionExtension.ext(asset)
    except pystac.errors.ExtensionNotImplemented:
        raise ValueError("No projection extension defined") from None

    if _proj.shape is None or _proj.transform is None or _proj.crs_string is None:
        raise ValueError(
            "The asset must have the following fields (from the projection extension):"
            " shape, transform, and one of an epsg, wkt2, or projjson"
        )

    h, w = _proj.shape
    if len(_proj.transform) not in (6, 9):
        raise ValueError("Asset transform must be 6 or 9 elements in size")

    if len(_proj.transform) == 9 and _proj.transform[-3:] != [0, 0, 1]:
        raise ValueError(f"Asset transform is not affine: {_proj.transform}")

    affine = Affine(*_proj.transform[:6])
    return GeoBox(wh_(w, h), affine, _proj.crs_string)


def geobox_gsd(geobox: GeoBox) -> float:
    """
    Compute ground sampling distance of a given GeoBox.

    :param geobox: input :class:`~datacube.utils.geometry.GeoBox`
    :returns: Minimum ground sampling distance along X/Y
    """
    return min(map(abs, [geobox.transform.a, geobox.transform.e]))  # type: ignore


def compute_eo3_grids(
    assets: Dict[str, pystac.asset.Asset]
) -> Tuple[Dict[str, GeoBox], Dict[str, str]]:
    """
    Compute a minimal set of eo3 grids.

    Pick default one, give names to non-default grids, while keeping track of
    which asset has which grid

    Assets must have ProjectionExtension with shape, transform and crs information
    populated.
    """
    # pylint: disable=too-many-locals

    assert len(assets) > 0

    def gbox_name(geobox: GeoBox) -> str:
        gsd = geobox_gsd(geobox)
        return f"g{gsd:g}"

    geoboxes = dicttoolz.valmap(asset_geobox, assets)

    # GeoBox to list of bands that share same footprint
    grids: Dict[GeoBox, List[str]] = {}
    crs: Optional[CRS] = None

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
        if crs is None:
            crs = grid.crs
        elif grid.crs != crs:
            raise ValueError("Expect all assets to share common CRS")

        grid_name = "default" if grid is g_default else gbox_name(grid)
        if grid_name in named_grids:
            raise NotImplementedError(
                "TODO: deal with multiple grids with same sampling distance"
            )

        named_grids[grid_name] = grid
        for band in bands:
            band2grid[band] = grid_name

    return named_grids, band2grid


def band2grid_from_gsd(assets: Dict[str, pystac.asset.Asset]) -> Dict[str, str]:
    grids: Dict[float, List[str]] = {}
    for name, asset in assets.items():
        gsd = asset.common_metadata.gsd
        gsd = 0 if gsd is None else gsd
        gsd_normed = float(f"{gsd:g}")
        grids.setdefault(gsd_normed, []).append(name)

    # Default grid is one with largest number of bands
    # .. and lowest gsd when there is a tie
    (_, default_gsd), *_ = sorted((-len(bands), gsd) for gsd, bands in grids.items())
    band2grid = {}
    for gsd, bands in grids.items():
        grid_name = "default" if gsd == default_gsd else f"g{gsd:g}"
        for band in bands:
            band2grid[band] = grid_name

    return band2grid


def alias_map_from_eo(item: pystac.item.Item, quiet: bool = False) -> Dict[str, str]:
    """
    Generate mapping ``common name -> canonical name`` for all unique common names defined on the Item eo extension.

    :param item: STAC :class:`~pystac.item.Item` to process
    :param quiet: Do not print warning if duplicate common names are found, defaults to False
    :return: common name to canonical name mapping
    """
    try:
        bands = EOExtension.ext(item, add_if_missing=False).bands
    except pystac.errors.ExtensionNotImplemented:
        return {}

    if bands is None:
        return {}  # pragma: no cover

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
    Create valid product name from an arbitrary string.

    Right now just maps ``-`` and `` `` to ``_``.

    :param name: Usually comes from ``collection_id``.
    """
    # TODO: for now just map `-`,` ` to `_`
    return name.replace("-", "_").replace(" ", "_")


def norm_band_metadata(
    v: Union[RasterBandMetadata, Dict[str, Any]]
) -> RasterBandMetadata:
    if isinstance(v, RasterBandMetadata):
        return v
    return RasterBandMetadata(
        v.get("data_type", "uint16"), v.get("nodata", 0), v.get("unit", "1")
    )


def mk_sample_item(collection: pystac.collection.Collection) -> pystac.item.Item:
    try:
        item_assets = ItemAssetsExtension.ext(collection).item_assets
    except pystac.errors.ExtensionNotImplemented:
        raise ValueError(
            "This only works on Collections with ItemAssets extension"
        ) from None

    item = pystac.item.Item(
        "sample",
        None,
        None,
        datetime.datetime(2020, 1, 1),
        {},
        stac_extensions=collection.stac_extensions,
        collection=collection,
    )

    for name, asset in item_assets.items():
        _asset = dict(href="")
        _asset.update(asset.to_dict())
        item.add_asset(name, pystac.asset.Asset.from_dict(_asset))

    return item
