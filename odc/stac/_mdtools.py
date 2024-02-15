"""
STAC -> EO3 utilities.

Utilities for translating STAC Items to EO3 Datasets.
"""

# pylint: disable=too-many-lines
from __future__ import annotations

import datetime
from collections import Counter
from copy import copy
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import pystac.asset
import pystac.collection
import pystac.errors
import pystac.item
import shapely.geometry
from affine import Affine
from odc.geo import (
    CRS,
    XY,
    Geometry,
    MaybeCRS,
    Resolution,
    SomeResolution,
    geom,
    res_,
    wh_,
    xy_,
)
from odc.geo.geobox import AnchorEnum, GeoBox, GeoboxAnchor
from odc.geo.math import maybe_zero, snap_scale, split_float
from odc.geo.types import Unset
from odc.geo.xr import ODCExtension
from pystac.extensions.eo import EOExtension
from pystac.extensions.item_assets import ItemAssetsExtension
from pystac.extensions.projection import ProjectionExtension
from pystac.extensions.raster import RasterBand, RasterExtension
from toolz import dicttoolz

from odc.loader.types import (
    MDParser,
    RasterBandMetadata,
    RasterGroupMetadata,
    RasterSource,
    norm_nodata,
    with_default,
)

from .model import (
    BandKey,
    BandQuery,
    MDParseConfig,
    ParsedItem,
    RasterCollectionMetadata,
)

ConversionConfig = Dict[str, Any]

EPSG4326 = CRS("EPSG:4326")

# Assets with these roles are ignored unless manually requested
ROLES_THUMBNAIL = {"thumbnail", "overview"}

# Used to detect image assets when media_type is missing
RASTER_FILE_EXTENSIONS = {
    "tif",
    "tiff",
    "jpeg",
    "jpg",
    "jp2",
    "img",
    "hdf",
    "nc",
    "zarr",
}

# image/* and these media-type are considered to be raster
NON_IMAGE_RASTER_MEDIA_TYPES = {
    "application/x-hdf",
    "application/x-hdf5",
    "application/hdf",
    "application/hdf5",
    "application/x-netcdf",
    "application/netcdf",
    "application/x-zarr",
    "application/zarr",
}


def _band_metadata_raw(asset: pystac.asset.Asset) -> List[RasterBand]:
    bands = asset.to_dict().get("raster:bands", None)
    if bands is None:
        return []
    return [RasterBand(props) for props in bands]


def band_metadata(
    asset: pystac.asset.Asset, default: RasterBandMetadata
) -> List[RasterBandMetadata]:
    """
    Compute band metadata from Asset raster extension with defaults from default.

    :param asset: Asset with raster extension
    :param default: Values to use for fallback
    :return: List of BandMetadata constructed from raster:bands metadata
    """
    bands: List[RasterBand] = []
    try:
        rext = RasterExtension.ext(asset)
        if rext.bands is not None:
            bands = rext.bands
    except pystac.errors.ExtensionNotImplemented:
        bands = _band_metadata_raw(asset)

    if len(bands) == 0:
        return [default]

    return [
        RasterBandMetadata(
            with_default(band.data_type, default.data_type),
            with_default(norm_nodata(band.nodata), default.nodata),
            with_default(band.unit, default.unit),
        )
        for band in bands
    ]


def has_proj_ext(item: Union[pystac.item.Item, pystac.collection.Collection]) -> bool:
    """
    Check if STAC Item or Collection has projection extension.

    :returns: ``True`` if PROJ extension is enabled
    :returns: ``False`` if no PROJ extension was found
    """
    if ProjectionExtension.has_extension(item):
        return True
    # can remove this block once pystac 1.9.0 is the min supported version
    return any(
        ext_name.startswith("https://stac-extensions.github.io/projection/")
        for ext_name in item.stac_extensions
    )


def has_raster_ext(item: Union[pystac.item.Item, pystac.collection.Collection]) -> bool:
    """
    Check if STAC Item/Collection have RasterExtension.

    :returns: ``True`` if Raster extension is enabled
    :returns: ``False`` if no Raster extension was found
    """
    if RasterExtension.has_extension(item):
        return True
    # can remove this block once pystac 1.9.0 is the min supported version
    return any(
        ext_name.startswith("https://stac-extensions.github.io/raster/")
        for ext_name in item.stac_extensions
    )


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
            and has_proj_ext(asset.owner)  # type: ignore
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

        ext = asset.href.split(".")[-1].lower()
        return ext in RASTER_FILE_EXTENSIONS

    media_type, *_ = media_type.split(";")
    media_type = media_type.lower()

    if media_type.startswith("image/"):
        # Image:
        #    False -- when thumbnail
        #    True  -- otherwise
        if any(r in roles for r in ROLES_THUMBNAIL):
            return False
        return True

    if media_type in NON_IMAGE_RASTER_MEDIA_TYPES:
        return True

    # some unsupported mime type
    return False


def mk_1x1_geobox(g: Geometry) -> GeoBox:
    """
    Construct 1x1 pixels GeoBox tightly enclosing supplied geometry.

    :param geom: Geometry in whatever projection
    :return: GeoBox object such that geobox.extent.contains(geom) is True, geobox.shape == (1,1)
    """
    x1, y1, x2, y2 = g.boundingbox
    # note that Y axis is inverted
    #   0,0 -> X_min, Y_max
    #   1,1 -> X_max, Y_min
    return GeoBox((1, 1), Affine((x2 - x1), 0, x1, 0, (y1 - y2), y2), g.crs)


def _gbox_anchor(gbox: GeoBox, tol: float = 1e-3) -> GeoboxAnchor:
    def _anchor(px: float, tol: float) -> float:
        _, x = split_float(px)  # x in (-0.5, +0.5)
        x = (1 + x) if x < 0 else x  # x in [0, 1)
        x = snap_scale(maybe_zero(x, tol), tol)
        return 0 if x >= 1 else x

    anchor = tuple(_anchor(px, tol) for px in (~gbox.transform) * (0, 0))
    if anchor == (0, 0):
        return AnchorEnum.EDGE
    if anchor == (0.5, 0.5):
        return AnchorEnum.CENTER
    return xy_(anchor)


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

    :param geobox: input :class:`~odc.geo.geobox.GeoBox`
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
    assert len(assets) > 0

    geoboxes = dicttoolz.valmap(asset_geobox, assets)
    return _group_geoboxes(geoboxes)


def _group_geoboxes(
    geoboxes: Dict[str, GeoBox]
) -> Tuple[Dict[str, GeoBox], Dict[str, str]]:
    # pylint: disable=too-many-locals
    assert len(geoboxes) > 0

    def gbox_name(geobox: GeoBox) -> str:
        gsd = geobox_gsd(geobox)
        return f"g{gsd:g}"

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

        grid_name = "default" if grid is g_default else gbox_name(grid)
        if grid_name in named_grids:
            band, *_ = bands
            grid_name = f"{grid_name}-{band}"

        named_grids[grid_name] = grid
        for band in bands:
            band2grid[band] = grid_name

    return named_grids, band2grid


def band2grid_from_gsd(assets: Dict[str, pystac.asset.Asset]) -> Dict[str, str]:
    if not assets:
        return {}

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


def _extract_aliases(
    asset_name: str,
    asset: pystac.asset.Asset,
    block_list: Set[str],
) -> Iterator[Tuple[str, int, BandKey]]:
    try:
        eo = EOExtension.ext(asset)
    except pystac.errors.ExtensionNotImplemented:
        return
    if eo.bands is None:
        return

    for idx, band in enumerate(eo.bands):
        for alias in [band.name, band.common_name]:
            if alias is not None and alias not in block_list:
                yield (alias, len(eo.bands), (asset_name, idx + 1))


def alias_map_from_eo(item: pystac.item.Item) -> Dict[str, List[BandKey]]:
    """
    Generate mapping ``common name -> canonical name``.

    For all unique common names defined on the Item's assets via the eo
    extension, record a mapping to the asset key ("canonical name"). Non-unique
    common names are ignored with a warning unless ``quiet`` flag is set.

    :param item: STAC :class:`~pystac.item.Item` to process
    :return: common name to (asset, idx) mapping
    """
    aliases: Dict[str, List[BandKey]] = {}

    asset_band_counts: Dict[str, int] = {}
    asset_names = set(item.assets)
    for asset_name, asset in item.assets.items():
        for alias, count, bkey in _extract_aliases(asset_name, asset, asset_names):
            aliases.setdefault(alias, []).append(bkey)
            asset_band_counts[asset_name] = count

    # Alias pointing to an asset with fewer bands is
    # of higher priority, 1-band data asset vs 3 band visual
    def _cmp(x):
        asset, _ = x
        return (asset_band_counts[asset], asset)

    return {alias: sorted(bands, key=_cmp) for alias, bands in aliases.items()}


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
        _asset = {"href": ""}
        _asset.update(asset.to_dict())
        item.add_asset(name, pystac.asset.Asset.from_dict(_asset))

    return item


def _collection_id(item: pystac.item.Item) -> str:
    # choose first that is set
    # 1. collection_id
    # 2. odc:product
    # 3. "_"
    if item.collection_id is None:
        # early ODC data
        return str(item.properties.get("odc:product", "_"))
    return str(item.collection_id)


class _CMDAssembler:
    """
    Incrementally build up collection metadata from item stream.

    Expect to see items of the same collection only.
    """

    # pylint: disable=too-few-public-methods,too-many-instance-attributes

    def __init__(
        self,
        collection_id: str,
        cfg: Optional[ConversionConfig] = None,
        md_plugin: MDParser | None = None,
    ) -> None:
        if cfg is None:
            cfg = {}

        self._cfg = MDParseConfig.from_dict(collection_id, cfg)
        self.check_proj: bool = not self._cfg.ignore_proj
        self.has_proj: Optional[bool] = None
        self.collection_id = collection_id
        self.md: Optional[RasterCollectionMetadata] = None
        self.md_plugin = md_plugin
        self._asset_keeps: Dict[str, bool] = {}
        self._known_assets: Set[str] = set()

    def _keep(self, kv: Tuple[str, pystac.asset.Asset]) -> bool:
        c = self._cfg
        name, asset = kv
        if name in c.band_cfg:
            return True
        assert self.has_proj is not None
        return is_raster_data(asset, check_proj=self.has_proj)

    def _extract_bands(
        self, name: str, asset: pystac.asset.Asset
    ) -> Dict[BandKey, RasterBandMetadata]:
        c = self._cfg
        bm = c.band_cfg.get(name, None)
        if bm is not None:
            return {(name, 1): copy(bm)}

        bm = c.band_cfg.get(f"{name}.*", None)
        if bm is None:
            bm = c.band_defaults

        bands = band_metadata(asset, bm)

        return {(name, idx + 1): bm for idx, bm in enumerate(bands)}

    def _bootstrap(self, item: pystac.item.Item):
        """Called on the very first item only."""
        self.has_proj = has_proj_ext(item) if self.check_proj else False
        if self.md_plugin is not None:
            md = self.md_plugin.extract(item)
            used_assets = set(n for n, _ in md.bands)
            data_bands = {n: item.assets[n] for n in used_assets}
        else:
            data_bands = dicttoolz.itemfilter(self._keep, item.assets)

            # found no data bands with check_proj=True
            # so try again with check_proj=False
            if len(data_bands) == 0 and self.has_proj:
                self.has_proj = False
                self.check_proj = False
                data_bands = dicttoolz.itemfilter(self._keep, item.assets)

            bands: Dict[BandKey, RasterBandMetadata] = {}
            aliases = alias_map_from_eo(item)

            # 1. If band in user config -- use that
            # 2. Use data from raster extension (with fallback to "*" config)
            # 3. Use config for "*" from user config as fallback
            for name, asset in data_bands.items():
                bands.update(self._extract_bands(name, asset))

            for alias, bkey in self._cfg.aliases.items():
                aliases.setdefault(alias, []).insert(0, bkey)
            md = RasterGroupMetadata(bands, aliases)

        # We assume that grouping of data bands into grids is consistent across
        # entire collection, so we compute it once and keep it
        if self.has_proj:
            _, band2grid = compute_eo3_grids(data_bands)
        else:
            band2grid = band2grid_from_gsd(data_bands)

        self._asset_keeps = {name: name in data_bands for name in item.assets}
        self._known_assets = set(self._asset_keeps)

        self.md = RasterCollectionMetadata(
            self.collection_id,
            md,
            has_proj=self.has_proj,
            band2grid=band2grid,
        )

    def update(self, item: pystac.item.Item):
        # pylint: disable=too-many-locals,too-many-branches
        if self.md is None:
            self._bootstrap(item)
            return

        new_assets = set(item.assets) - self._known_assets
        if len(new_assets) == 0:
            return

        new_data_assets: List[Tuple[str, pystac.asset.Asset]] = []
        for name in new_assets:
            asset = item.assets[name]
            is_data = self._keep((name, asset))
            self._asset_keeps[name] = is_data
            if is_data:
                new_data_assets.append((name, asset))
        self._known_assets = set(self._asset_keeps)

        # some new assets that we don't care about
        if len(new_data_assets) == 0:
            return

        bands = self.md.meta.bands
        aliases = self.md.meta.aliases
        band2grid = self.md.band2grid

        # GeoBox -> grid name
        grid2band: Dict[GeoBox, str] = {}
        if self.has_proj:
            for name, asset in item.assets.items():
                if (grid_name := band2grid.get(name, None)) is not None:
                    grid2band[asset_geobox(asset)] = grid_name

        for name, asset in new_data_assets:
            bands.update(self._extract_bands(name, asset))

            # update alias table
            for alias, count, bkey in _extract_aliases(name, asset, self._known_assets):
                _bands = aliases.setdefault(alias, [])
                if count == 1:
                    _bands.insert(0, bkey)
                else:
                    _bands.append(bkey)

            if self.has_proj:
                band2grid[name] = grid2band.get(asset_geobox(asset), f"grid-{name}")


def extract_collection_metadata(
    item: pystac.item.Item,
    cfg: Optional[ConversionConfig] = None,
    md_plugin: MDParser | None = None,
) -> RasterCollectionMetadata:
    """
    Use sample item to figure out raster bands within the collection.

    1. Decide which assets contain raster data
    2. Extract metadata about about rasters from STAC or from ``cfg``
    3. See if ``proj`` data is available and group bands by resolution
    4. Construct alias map from common names and user config

    :param item: Representative STAC item from a collection.
    :param cfg: Optional user configuration
    :return: :py:class:`~odc.stac._model.RasterCollectionMetadata`
    """
    collection_id = _collection_id(item)
    proc = _CMDAssembler(collection_id, cfg, md_plugin)
    proc.update(item)
    assert proc.md is not None
    return proc.md


def parse_item(
    item: pystac.item.Item,
    template: Union[RasterCollectionMetadata, ConversionConfig, None] = None,
    md_plugin: MDParser | None = None,
) -> ParsedItem:
    """
    Extract raster band information relevant for data loading.

    :param item: STAC Item
    :param template: Common collection level information
    :return: ``ParsedItem``
    """
    # pylint: disable=too-many-locals
    if not isinstance(template, RasterCollectionMetadata):
        template = extract_collection_metadata(item, template, md_plugin)

    band2grid = template.band2grid
    has_proj = False if template.has_proj is False else has_proj_ext(item)
    _assets = item.assets

    _grids: Dict[str, GeoBox] = {}
    bands: Dict[BandKey, RasterSource] = {}
    geometry: Optional[Geometry] = None

    if item.geometry is not None:
        geometry = Geometry(item.geometry, EPSG4326)

    def _get_grid(grid_name: str, asset: pystac.asset.Asset) -> GeoBox:
        grid = _grids.get(grid_name, None)
        if grid is not None:
            return grid
        grid = asset_geobox(asset)
        _grids[grid_name] = grid
        return grid

    for bk, meta in template.meta.bands.items():
        asset_name, band_idx = bk
        asset = _assets.get(asset_name)
        if asset is None:
            continue

        grid_name = band2grid.get(asset_name, "default")
        geobox: Optional[GeoBox] = _get_grid(grid_name, asset) if has_proj else None

        uri = asset.get_absolute_href()
        if uri is None:
            raise ValueError(
                f"Can not determine absolute path for band: {asset_name}"
            )  # pragma: no cover (https://github.com/stac-utils/pystac/issues/754)

        driver_data: Any = None
        if md_plugin is not None:
            driver_data = md_plugin.driver_data(asset, bk)

        bands[bk] = RasterSource(
            uri=uri,
            band=band_idx,
            geobox=geobox,
            meta=meta,
            driver_data=driver_data,
        )

    md = item.common_metadata
    return ParsedItem(
        item.id,
        template,
        bands,
        geometry,
        datetime=item.datetime,
        datetime_range=(md.start_datetime, md.end_datetime),
        href=item.get_self_href(),
    )


def parse_items(
    items: Iterable[pystac.item.Item],
    cfg: Optional[ConversionConfig] = None,
    md_plugin: MDParser | None = None,
) -> Iterator[ParsedItem]:
    """
    Parse sequence of STAC Items into internal representation.

    Exposed for debugging purposes.
    """
    proc_cache: Dict[str, _CMDAssembler] = {}

    for item in items:
        collection_id = _collection_id(item)
        proc = proc_cache.get(collection_id, None)
        if proc is None:
            proc = _CMDAssembler(collection_id, cfg, md_plugin=md_plugin)
            proc_cache[collection_id] = proc

        proc.update(item)
        yield parse_item(item, proc.md, md_plugin)


def _most_common_gbox(
    gboxes: Sequence[GeoBox],
    thresh: float = 0.1,
) -> Tuple[Optional[CRS], Resolution, GeoboxAnchor, Optional[GeoBox]]:
    gboxes = list(gboxes)

    # First check for identical geoboxes
    _gboxes = set(gboxes)
    if len(_gboxes) == 1:
        g = _gboxes.pop()
        return (g.crs, g.resolution, _gbox_anchor(g), g)

    # Most common shared CRS, Resolution, Anchor
    gg = [(g.crs, g.resolution, _gbox_anchor(g)) for g in gboxes]
    hist = Counter(gg)
    (best, n), *_ = hist.most_common(1)
    if n / len(gg) > thresh:
        return (*best, None)

    # too few in the majority group
    # redo ignoring anchor this time
    hist = Counter((crs, res) for crs, res, _ in gg)
    (best, _), *_ = hist.most_common(1)
    return (*best, AnchorEnum.EDGE, None)


def _auto_load_params(
    items: Sequence[ParsedItem], bands: Optional[Sequence[str]] = None
) -> Optional[Tuple[Optional[CRS], Resolution, GeoboxAnchor, Optional[GeoBox]]]:
    def _extract_gbox(
        item: ParsedItem,
    ) -> Optional[GeoBox]:
        gbx = item.geoboxes(bands)
        return gbx[0] if len(gbx) else None

    geoboxes = [gbox for gbox in map(_extract_gbox, items) if gbox is not None]
    if len(geoboxes) == 0:
        return None

    return _most_common_gbox(geoboxes, 0.1)


def _normalize_geometry(xx: Any) -> Geometry:
    if isinstance(xx, shapely.geometry.base.BaseGeometry):
        return Geometry(xx, "epsg:4326")

    if isinstance(xx, Geometry):
        return xx

    if isinstance(xx, dict):
        return Geometry(xx, "epsg:4326")

    # GeoPandas
    _geo = getattr(xx, "__geo_interface__", None)
    if _geo is None:
        raise ValueError("Can't interpret value as geometry")

    _crs = getattr(xx, "crs", "epsg:4326")
    return Geometry(_geo, _crs)


def _compute_bbox(
    items: Iterable[ParsedItem],
    crs: MaybeCRS,
    bands: BandQuery = None,
) -> geom.BoundingBox:
    def bboxes(items: Iterable[ParsedItem]) -> Iterator[geom.BoundingBox]:
        crs0 = crs
        for item in items:
            g = item.safe_geometry(crs0, bands=bands)
            assert g is not None
            if crs0 is crs:
                # If crs is something like "utm", make sure
                # same one is used going forward
                crs0 = g.crs
            yield g.boundingbox

    return geom.bbox_union(bboxes(items))


def _align2anchor(
    align: Optional[Union[float, int, XY[float]]], resolution: SomeResolution
) -> GeoboxAnchor:
    if align is None:
        return AnchorEnum.EDGE

    if isinstance(align, (float, int)):
        align = xy_(align, align)

    # support old-style "align", which is basically anchor but in CRS units
    ax, ay = align.xy
    if ax == 0 and ay == 0:
        return AnchorEnum.EDGE
    resolution = res_(resolution)
    return xy_(ax / abs(resolution.x), ay / abs(resolution.y))


def output_geobox(
    items: Sequence[ParsedItem],
    bands: Optional[Sequence[str]] = None,
    *,
    crs: MaybeCRS = Unset(),
    resolution: Optional[SomeResolution] = None,
    anchor: Optional[GeoboxAnchor] = None,
    align: Optional[Union[float, int, XY[float]]] = None,
    geobox: Optional[GeoBox] = None,
    like: Optional[Any] = None,
    geopolygon: Optional[Any] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    lon: Optional[Tuple[float, float]] = None,
    lat: Optional[Tuple[float, float]] = None,
    x: Optional[Tuple[float, float]] = None,
    y: Optional[Tuple[float, float]] = None,
) -> Optional[GeoBox]:
    """
    Used to compute output geobox from load parameters.

    Exposed at top-level for debugging.
    """
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    # pylint: disable=too-many-return-statements,too-many-arguments

    # geobox, like --> GeoBox
    # lon,lat      --> geopolygon[epsg:4326]
    # bbox         --> geopolygon[epsg:4326]
    # x,y,crs      --> geopolygon[crs]
    # [items]      --> crs, geopolygon[crs]
    # [items]      --> crs, resolution
    # geopolygon, crs, resolution[, anchor|align] --> GeoBox

    params = {
        k
        for k, v in {
            "x": x,
            "y": y,
            "lon": lon,
            "lat": lat,
            "crs": crs,
            "resolution": resolution,
            "align": align,
            "anchor": anchor,
            "like": like,
            "geopolygon": geopolygon,
            "bbox": bbox,
            "geobox": geobox,
        }.items()
        if not (v is None or isinstance(v, Unset))
    }

    def report_extra_args(primary: str, *ok_args):
        args = params - set([primary, *ok_args])
        if len(args) > 0:
            raise ValueError(
                f"Too many arguments when using `{primary}=`: {','.join(args)}"
            )

    def check_arg_sets(*args: str) -> bool:
        x = params & set(args)
        if len(x) == 0 or len(x) == len(args):
            return True
        return False

    if geobox is not None:
        report_extra_args("geobox")
        return geobox
    if like is not None:
        report_extra_args("like")
        if isinstance(like, GeoBox):
            return like
        _odc = getattr(like, "odc", None)
        if _odc is None:
            raise ValueError("No geospatial info on `like=` input")

        assert isinstance(_odc, ODCExtension)
        if _odc.geobox is None:
            raise ValueError("No geospatial info on `like=` input")

        assert isinstance(_odc.geobox, GeoBox)
        return _odc.geobox

    if not check_arg_sets("x", "y"):
        raise ValueError("Need to supply both x= and y=")

    if not check_arg_sets("lon", "lat"):
        raise ValueError("Need to supply both lon= and lat=")

    if isinstance(crs, Unset):
        crs = None

    grid_params = ("crs", "align", "anchor", "resolution")

    query_crs: Optional[CRS] = None
    if geopolygon is not None:
        geopolygon = _normalize_geometry(geopolygon)
        query_crs = geopolygon.crs

    # Normalize  x.y|lon.lat|bbox|geopolygon arguments to a geopolygon|None
    if geopolygon is not None:
        report_extra_args("geopolygon", *grid_params)
    elif bbox is not None:
        report_extra_args("bbox", *grid_params)
        x0, y0, x1, y1 = bbox
        geopolygon = geom.box(x0, y0, x1, y1, EPSG4326)
    elif lat is not None and lon is not None:
        # lon=(x0, x1), lat=(y0, y1)
        report_extra_args("lon,lat", "lon", "lat", *grid_params)
        x0, x1 = sorted(lon)
        y0, y1 = sorted(lat)
        geopolygon = geom.box(x0, y0, x1, y1, EPSG4326)
    elif x is not None and y is not None:
        if crs is None:
            raise ValueError("Need to supply `crs=` when using `x=`, `y=`.")
        report_extra_args("x,y", "x", "y", *grid_params)
        x0, x1 = sorted(x)
        y0, y1 = sorted(y)
        geopolygon = geom.box(x0, y0, x1, y1, crs)

    full_auto = len(params) == 0
    _anchor: GeoboxAnchor = AnchorEnum.EDGE
    _the_gbox: Optional[GeoBox] = None

    if crs is None or resolution is None:
        rr = _auto_load_params(items, bands)
        if rr is not None:
            _crs, _res, _anchor, _the_gbox = rr
        else:
            _crs, _res = None, None

        if full_auto and _the_gbox is not None:
            return _the_gbox

        if crs is None:
            crs = _crs or query_crs

        if resolution is None:
            resolution = _res

        if resolution is None or crs is None:
            return None

    if anchor is None:
        anchor = _anchor if align is None else _align2anchor(align, resolution)

    if geopolygon is not None:
        assert isinstance(geopolygon, Geometry)
        return GeoBox.from_geopolygon(
            geopolygon,
            resolution=resolution,
            crs=crs,
            anchor=anchor,
        )

    # compute from parsed items
    _bbox = _compute_bbox(items, crs)
    return GeoBox.from_bbox(_bbox, resolution=resolution, anchor=anchor)
