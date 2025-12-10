"""Metadata and data loading model classes."""

from __future__ import annotations

import datetime as dt
import math
from copy import copy
from dataclasses import astuple, dataclass, field, replace
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import numpy as np
from odc.geo import CRS, Geometry, MaybeCRS
from odc.geo.geobox import GeoBox
from odc.geo.types import Unset
from odc.loader.types import (
    AuxBandMetadata,
    AuxDataSource,
    BandIdentifier,
    BandKey,
    BandQuery,
    FixedCoord,
    RasterBandMetadata,
    RasterGroupMetadata,
    RasterSource,
    norm_band_metadata,
    norm_key,
)
from typing_extensions import override


@dataclass(eq=True, frozen=True)
class RasterCollectionMetadata(
    Mapping[BandIdentifier, RasterBandMetadata | AuxBandMetadata]
):
    """
    Information about raster data in a collection.

    We assume that assets with the same names have the same kind of raster data across items within
    a collection. This is built from the combination of data collected from STAC and user
    configuration if supplied.
    """

    name: str
    """Collection name."""

    meta: RasterGroupMetadata
    """
    Band, aliases and extra dimensions metadata.
    """

    has_proj: bool
    """
    Whether to expect/look for ``proj`` extension on item assets.

    Proj data extraction can be disabled by the user with config. It is also disabled if it was not
    detected in the first item.
    """

    band2grid: Dict[str, str]
    """
    Band name to grid name mapping.

    Bands that share the same geometry map to the same grid name. Usually all bands share one common
    grid with the name ``default``. Here again we assume that this grouping of bands to grids is
    stable across the entire collection. This information is used to decide default projection and
    resolution at load time.

    Right now grid information is only extracted from STAC, so any savings from looking up this
    information once across all bands that share common grid is relatively insignificant, but if we
    ever support looking that up from the actual raster data this can speed up the process. This
    also reduces memory pressure somewhat as many bands will share one grid object.
    """

    def band_aliases(self, unique: bool = False) -> Dict[BandKey, List[str]]:
        """
        Compute inverse of alias mapping.

        :return:
          Mapping from canonical name to a list of defined aliases.
        """
        out: Dict[BandKey, List[str]] = {}
        for alias, canon_names in self.meta.aliases.items():
            if unique:
                canon_names = canon_names[:1]

            for cn in canon_names:
                out.setdefault(cn, []).append(alias)
        return out

    def _norm_key(self, k: BandKey) -> str:
        asset, idx = k

        # if single band asset it's just asset name
        if idx == 1 and (asset, 2) not in self.meta.bands and asset != "_stac_metadata":
            return asset

        # if any alias references this key as first choice return that
        for alias, (_k, *_) in self.meta.aliases.items():
            if _k == k:
                return alias

        # Finaly use . notation
        return f"{asset}.{idx}"

    @property
    def all_bands(self) -> List[str]:
        return [self._norm_key(k) for k in self.meta.bands]

    @property
    def prop_bands(self) -> List[str]:
        return [self._norm_key(k) for k in self.meta.bands if k[0] == "_stac_metadata"]

    def normalize_band_query(self, bands: BandQuery = None) -> List[str]:
        if bands is None:
            return self.all_bands

        if isinstance(bands, str):
            bands = [bands]
        elif not isinstance(bands, list):
            bands = list(bands)

        # when subset of raster bands is requested, we still add properties to
        # the query, unless query references at least one property also
        _props = self.prop_bands
        if any(b in _props for b in bands):
            return bands
        return bands + _props

    def resolve_bands(
        self,
        bands: BandQuery = None,
    ) -> Dict[str, RasterBandMetadata | AuxBandMetadata]:
        """
        Query bands taking care of aliases.
        """
        query = self.normalize_band_query(bands)

        return {
            band: self.meta.bands[k]
            for band, k in ((band, self.band_key(band)) for band in query)
        }

    def band_key(self, band: str) -> BandKey:
        """
        Compute canonical band key for an alias/band.

        ``(asset name: str,  band index: int 1..)``
        """
        if (band, 1) in self.meta.bands:
            return (band, 1)

        candidates = self.meta.aliases.get(band, [])
        n = len(candidates)
        if n == 1:
            return candidates[0]
        if n > 1:
            # maybe warn about ambiguity?
            return candidates[0]

        # check if it's asset.<index> form
        parts = band.rsplit(".", 1)
        if len(parts) > 1:
            band, idx = parts
            return (band, int(idx))

        raise ValueError(f"No such band/alias: {band}")

    def canonical_name(self, band: str) -> str:
        """
        Canonical name for an alias.
        """
        return self._norm_key(self.band_key(band))

    @override
    def __getitem__(self, band: BandIdentifier) -> RasterBandMetadata | AuxBandMetadata:
        """
        Query band taking care of aliases.

        :raises: :py:class:`KeyError`
        """
        if isinstance(band, str):
            try:
                band = self.band_key(band)
            except ValueError:
                raise KeyError(band) from None
        return self.meta.bands[band]

    @property
    def bands(self) -> Mapping[BandKey, RasterBandMetadata | AuxBandMetadata]:
        return self.meta.bands

    def meta_for(self, bands: BandQuery = None) -> RasterGroupMetadata:
        """
        Extract raster group metadata for a subset of bands.

        Output uses supplied band names as keys, effectively replacing canonical
        names with aliases supplied by the user.
        """
        return self.meta.patch(
            bands={norm_key(b): self[b] for b in self.normalize_band_query(bands)}
        )

    @property
    def aliases(self) -> Dict[str, List[BandKey]]:
        return self.meta.aliases

    @override
    def __len__(self) -> int:
        return len(self.meta.bands)

    @override
    def __iter__(self) -> Iterator[BandKey]:
        yield from self.meta.bands

    @override
    def __contains__(self, __o: object) -> bool:
        if isinstance(__o, tuple):
            return __o in self.meta.bands
        if isinstance(__o, str):
            return __o in self.meta.aliases or norm_key(__o) in self.meta.bands
        return False

    def __dask_tokenize__(self):
        return astuple(self)

    def patch(self, **kwargs) -> "RasterCollectionMetadata":
        return replace(self, **kwargs)

    def asset_names(self) -> tuple[str, ...]:
        out: list[str] = []
        seen: set[str] = set()
        for asset_name, _ in self.meta.bands:
            if asset_name != "_stac_metadata":
                if asset_name in seen:
                    continue
                seen.add(asset_name)
                out.append(asset_name)

        return tuple(out)


@dataclass(eq=True, frozen=True)
class ParsedItem(Mapping[BandIdentifier, RasterSource | AuxDataSource]):
    """
    Captures essentials parts for data loading from a STAC Item.

    Only includes raster bands of interest.
    """

    # pylint: disable=too-many-instance-attributes

    id: str
    """Item id copied from STAC."""

    collection: RasterCollectionMetadata
    """Collection this Item is part of."""

    bands: Mapping[BandKey, RasterSource | AuxDataSource]
    """Raster bands."""

    geometry: Optional[Geometry] = None
    """Footprint of the dataset."""

    datetime: Optional[dt.datetime] = None
    """Nominal timestamp."""

    datetime_range: Tuple[Optional[dt.datetime], Optional[dt.datetime]] = None, None
    """Time period covered."""

    href: Optional[str] = None
    """Self link from stac item."""

    accessories: dict[str, Any] = field(default_factory=dict)
    """Additional assets"""

    def geoboxes(self, bands: BandQuery = None) -> Tuple[GeoBox, ...]:
        """
        Unique ``GeoBox`` s, highest resolution first.

        :param bands: which bands to consider, default is all
        """
        bands = self.collection.normalize_band_query(bands)

        def _resolution(g: GeoBox) -> float:
            return min(g.resolution.map(abs).xy)

        # TODO: support other geobox types?
        gbx: Set[GeoBox] = set()
        for name in bands:
            b = self.bands.get(self.collection.band_key(name), None)
            if isinstance(b, RasterSource):
                if b.geobox is not None:
                    assert isinstance(b.geobox, GeoBox)
                    gbx.add(b.geobox)

        return tuple(sorted(gbx, key=_resolution))

    def crs(self, bands: BandQuery = None) -> Optional[CRS]:
        """
        First non-null CRS across assets.
        """
        for gbox in self.geoboxes(bands):
            if gbox.crs is not None:
                return gbox.crs

        return None

    def image_geometry(
        self,
        crs: MaybeCRS = Unset(),
        bands: BandQuery = None,
    ) -> Optional[Geometry]:
        """
        Extract footprint of a given band(s) from proj metadata in a given projection.
        """
        if isinstance(crs, Unset):
            crs = None

        for gbox in self.geoboxes(bands):
            if gbox.crs is not None:
                if crs is None or crs == gbox.crs:
                    return gbox.extent
                return gbox.footprint(crs)

        return None

    def safe_geometry(
        self,
        crs: MaybeCRS = Unset(),
        bands: BandQuery = None,
    ) -> Optional[Geometry]:
        """
        Get item geometry footprint in desired projection or native.

        1. Use full-image footprint if proj data is available
        2. Fallback to item geometry if not
        """

        img_geom = self.image_geometry(crs, bands=bands)
        if img_geom is not None:
            return img_geom

        if self.geometry is None:
            return None

        if crs is None or isinstance(crs, Unset):
            return self.geometry

        N = 100  # minimum number of points along perimiter we desire
        min_sample_distance = math.sqrt(self.geometry.area) * 4 / N
        return self.geometry.to_crs(
            crs,
            min_sample_distance,
            check_and_fix=True,
        ).dropna()

    def resolve_bands(
        self, bands: BandQuery = None
    ) -> dict[str, RasterSource | AuxDataSource | None]:
        """
        Query bands taking care of aliases.
        """
        bands = self.collection.normalize_band_query(bands)
        canon = self.collection.band_key

        return {
            k: self.bands.get(_actual, None)
            for k, _actual in ((k, canon(k)) for k in bands)
        }

    @override
    def __getitem__(self, band: BandIdentifier) -> RasterSource | AuxDataSource:
        """
        Query band taking care of aliases.

        :raises: :py:class:`KeyError`
        """
        if isinstance(band, str):
            band = self.collection.band_key(band)
        return self.bands[band]

    @override
    def __len__(self) -> int:
        return len(self.bands)

    @override
    def __iter__(self) -> Iterator[BandKey]:
        yield from self.bands

    @override
    def __contains__(self, k: object) -> bool:
        if isinstance(k, str):
            try:
                return self.collection.band_key(k) in self.bands
            except ValueError:
                return False
        if isinstance(k, tuple):
            return k in self.bands
        return False

    @property
    def nominal_datetime(self) -> dt.datetime:
        """
        Resolve timestamp to a single value.

        - datetime if set
        - start_datetime if set
        - end_datetime if set
        - ``raise ValueError`` otherwise
        """
        for ts in [self.datetime, *self.datetime_range]:
            if ts is not None:
                return ts
        raise ValueError("Timestamp was not populated.")

    @property
    def mid_longitude(self) -> Optional[float]:
        """
        Return longitude of the center point.

        used for "solar day" computation.
        """
        if self.geometry is None:
            return None
        ((lon, _),) = self.geometry.centroid.to_crs("epsg:4326").points
        return lon

    @property
    def solar_date(self) -> dt.datetime:
        """
        Nominal datetime adjusted by longitude.
        """
        lon = self.mid_longitude
        if lon is None:
            return self.nominal_datetime
        return _convert_to_solar_time(self.nominal_datetime, lon)

    def solar_date_at(self, lon: float) -> dt.datetime:
        """
        Nominal datetime adjusted by longitude.
        """
        return _convert_to_solar_time(self.nominal_datetime, lon)

    def strip(self) -> "ParsedItem":
        """
        Copy of self but with stripped bands.
        """
        return replace(
            self,
            bands={k: band.strip() for k, band in self.bands.items()},
            accessories={},
        )

    def assets(self) -> Dict[str, List[RasterSource]]:
        """
        Extract bands grouped by asset they belong to.
        """
        assets: Dict[str, List[Tuple[int, RasterSource]]] = {}
        for (asset, idx), src in self.bands.items():
            if isinstance(src, RasterSource):
                assets.setdefault(asset, []).append((idx, src))

        return {
            k: [src for _, src in sorted(srcs, key=lambda x: x[0])]
            for k, srcs in assets.items()
        }

    @override
    def __hash__(self) -> int:
        return hash((self.id, self.collection.name))

    def __dask_tokenize__(self):
        return (
            self.id,
            self.collection,
            self.bands,
            self.href,
            self.datetime,
            self.datetime_range,
        )


def _default_props_fuser(xx: Sequence[Any]) -> Any:
    n = len(xx)
    if n == 0:
        return None
    if n == 1:
        return xx[0]
    if isinstance(xx[0], str):
        return ",".join((str(x) for x in xx))

    xx = [x for x in xx if isinstance(x, (int, float)) and math.isfinite(x)]
    if len(xx) == 0:
        return None
    if len(xx) == 1:
        return xx[0]
    return sum(xx) / len(xx)


@dataclass(frozen=True)
class PropertyLoadRequest:
    """
    Request to load a property from STAC item as xarray DataArray.

    Attributes:
        key: The key of the property to load from STAC item
        name: Name to use for output DataArray, if None will use the property key
        dtype: Data type to use for loaded data, defaults to float32
    """

    key: str
    name: str | None = None
    dtype: str = "float32"
    nodata: float | None = None
    units: str = "1"
    fuser: Callable[[Sequence[Any]], Any] = _default_props_fuser

    @staticmethod
    def from_user_input(
        inputs: Sequence[str | Mapping[str, Any]],
    ) -> list["PropertyLoadRequest"]:
        """
        Create a list of PropertyLoadRequest objects from user input.

        Args:
            inputs: Sequence of either strings (property keys) or dictionaries with configuration.
                   Dictionaries must have 'key' defined, and can optionally have 'dtype' and 'name'.

        Returns:
            List of PropertyLoadRequest objects

        Raises:
            ValueError: If a dictionary input is missing the required 'key' field
        """

        def _norm(what: str | Mapping[str, Any]) -> "PropertyLoadRequest":
            if isinstance(what, str):
                return PropertyLoadRequest(key=what)
            if isinstance(what, dict):
                if "key" not in what:
                    raise ValueError("Dictionary input must contain 'key' field")
                return PropertyLoadRequest(**what)
            raise ValueError(f"Input must be string or dict, got {type(what)}")

        return [_norm(what) for what in inputs]

    @property
    def output_name(self) -> str:
        if self.name is not None:
            return self.name
        return self.key.replace(".", "_").replace(":", "_").replace("-", "_")

    @property
    def fill_value(self) -> Any:
        dtype = np.dtype(self.dtype)
        if self.nodata is not None:
            return dtype.type(self.nodata)
        if dtype.kind == "f":
            return dtype.type(float("nan"))
        return dtype.type(0)


@dataclass(frozen=True)
class MDParseConfig:
    """Item parsing config."""

    band_defaults: RasterBandMetadata = field(
        default_factory=lambda: norm_band_metadata({})
    )
    band_cfg: Dict[str, RasterBandMetadata] = field(default_factory=dict)
    aliases: Dict[str, BandKey] = field(default_factory=dict)
    ignore_proj: bool = False
    extra_dims: Dict[str, int] = field(default_factory=dict)
    extra_coords: Sequence[FixedCoord] = ()
    with_props: Sequence[PropertyLoadRequest] = field(default_factory=list)

    @staticmethod
    def from_dict(
        cfg: Dict[str, Any], collection_id: str | None = None
    ) -> "MDParseConfig":
        if collection_id is not None:
            if "assets" in cfg:  # Assume it's a single collection config
                _cfg = copy(cfg)
            else:
                _cfg = copy(cfg.get("*", {}))
                _cfg.update(cfg.get(collection_id, {}))
        else:
            _cfg = copy(cfg)

        band_defaults, band_cfg = _norm_band_cfg(_cfg.get("assets", {}))

        aliases = {
            alias: ((band, 1) if isinstance(band, str) else band)
            for alias, band in _cfg.get("aliases", {}).items()
        }
        ignore_proj: bool = _cfg.get("ignore_proj", False)
        extra_dims: Dict[str, int] = _cfg.get("dims", {})
        extra_coords: list[FixedCoord] = []
        cc: dict[str, list[Any]] = _cfg.get("coords", {})
        assert isinstance(cc, dict)
        with_props = _cfg.get("with_properties", cfg.get("with_properties", []))
        assert isinstance(with_props, list)

        for name, val in cc.items():
            assert isinstance(val, list)
            extra_coords.append(FixedCoord(name, val))

        return MDParseConfig(
            band_defaults=band_defaults,
            band_cfg=band_cfg,
            ignore_proj=ignore_proj,
            aliases=aliases,
            extra_dims=extra_dims,
            extra_coords=tuple(extra_coords),
            with_props=PropertyLoadRequest.from_user_input(with_props),
        )


def _norm_band_cfg(
    cfg: Dict[str, Any],
) -> Tuple[RasterBandMetadata, Dict[str, RasterBandMetadata]]:
    fallback = norm_band_metadata(cfg.get("*", {}))
    return fallback, {
        k: norm_band_metadata(v, fallback) for k, v in cfg.items() if k != "*"
    }


def _convert_to_solar_time(utc: dt.datetime, longitude: float) -> dt.datetime:
    # offset_seconds snapped to 1 hour increments
    #    1/15 == 24/360 (hours per degree of longitude)
    offset_seconds = int(longitude / 15) * 3600
    return utc + dt.timedelta(seconds=offset_seconds)
