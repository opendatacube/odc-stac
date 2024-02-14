"""Metadata and data loading model classes."""

import datetime as dt
import math
from copy import copy
from dataclasses import astuple, dataclass, field, replace
from typing import Any, Dict, Iterator, List, Mapping, Optional, Set, Tuple

from odc.geo import CRS, Geometry, MaybeCRS
from odc.geo.geobox import GeoBox
from odc.geo.types import Unset

from odc.loader.types import (
    BandIdentifier,
    BandKey,
    BandQuery,
    RasterBandMetadata,
    RasterSource,
    norm_band_metadata,
    norm_key,
)


@dataclass(eq=True, frozen=True)
class RasterCollectionMetadata(Mapping[BandIdentifier, RasterBandMetadata]):
    """
    Information about raster data in a collection.

    We assume that assets with the same names have the same kind of raster data across items within
    a collection. This is built from the combination of data collected from STAC and user
    configuration if supplied.
    """

    name: str
    """Collection name."""

    bands: Dict[BandKey, RasterBandMetadata]
    """
    Bands are assets that contain raster data.

    This controls which assets are extracted from STAC.
    """

    aliases: Dict[str, List[BandKey]]
    """
    Alias map ``alias -> [(asset, idx),...]``.

    Used to rename bands at load time.
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
        for alias, canon_names in self.aliases.items():
            if unique:
                canon_names = canon_names[:1]

            for cn in canon_names:
                out.setdefault(cn, []).append(alias)
        return out

    def _norm_key(self, k: BandKey) -> str:
        asset, idx = k

        # if single band asset it's just asset name
        if idx == 1 and (asset, 2) not in self.bands:
            return asset

        # if any alias references this key as first choice return that
        for alias, (_k, *_) in self.aliases.items():
            if _k == k:
                return alias

        # Finaly use . notation
        return f"{asset}.{idx}"

    @property
    def all_bands(self) -> List[str]:
        return [self._norm_key(k) for k in self.bands]

    def normalize_band_query(self, bands: BandQuery = None) -> List[str]:
        if isinstance(bands, str):
            return [bands]
        if bands is None:
            return self.all_bands
        return list(bands)

    def resolve_bands(self, bands: BandQuery = None) -> Dict[str, RasterBandMetadata]:
        """
        Query bands taking care of aliases.
        """
        bands = self.normalize_band_query(bands)
        return {
            band: self.bands[k]
            for band, k in ((band, self.band_key(band)) for band in bands)
        }

    def band_key(self, band: str) -> BandKey:
        """
        Compute canonical band key for an alias/band.

        ``(asset name: str,  band index: int 1..)``
        """
        if (band, 1) in self.bands:
            return (band, 1)

        candidates = self.aliases.get(band, [])
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

    def __getitem__(self, band: BandIdentifier) -> RasterBandMetadata:
        """
        Query band taking care of aliases.

        :raises: :py:class:`KeyError`
        """
        if isinstance(band, str):
            try:
                band = self.band_key(band)
            except ValueError:
                raise KeyError(band) from None
        return self.bands[band]

    def __len__(self) -> int:
        return len(self.bands)

    def __iter__(self) -> Iterator[BandKey]:
        yield from self.bands

    def __contains__(self, __o: object) -> bool:
        if isinstance(__o, tuple):
            return __o in self.bands
        if isinstance(__o, str):
            return __o in self.aliases or norm_key(__o) in self.bands
        return False

    def __dask_tokenize__(self):
        return astuple(self)


@dataclass(eq=True, frozen=True)
class ParsedItem(Mapping[BandIdentifier, RasterSource]):
    """
    Captures essentials parts for data loading from a STAC Item.

    Only includes raster bands of interest.
    """

    id: str
    """Item id copied from STAC."""

    collection: RasterCollectionMetadata
    """Collection this Item is part of."""

    bands: Dict[BandKey, RasterSource]
    """Raster bands."""

    geometry: Optional[Geometry] = None
    """Footprint of the dataset."""

    datetime: Optional[dt.datetime] = None
    """Nominal timestamp."""

    datetime_range: Tuple[Optional[dt.datetime], Optional[dt.datetime]] = None, None
    """Time period covered."""

    href: Optional[str] = None
    """Self link from stac item."""

    def geoboxes(self, bands: BandQuery = None) -> Tuple[GeoBox, ...]:
        """
        Unique ``GeoBox`` s, highest resolution first.

        :param bands: which bands to consider, default is all
        """
        bands = self.collection.normalize_band_query(bands)

        def _resolution(g: GeoBox) -> float:
            return min(g.resolution.map(abs).xy)  # type: ignore

        gbx: Set[GeoBox] = set()
        for name in bands:
            b = self.bands.get(self.collection.band_key(name), None)
            if b is not None:
                if b.geobox is not None:
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
    ) -> Dict[str, Optional[RasterSource]]:
        """
        Query bands taking care of aliases.
        """
        bands = self.collection.normalize_band_query(bands)
        canon = self.collection.band_key

        return {
            k: self.bands.get(_actual, None)
            for k, _actual in ((k, canon(k)) for k in bands)
        }

    def __getitem__(self, band: BandIdentifier) -> RasterSource:
        """
        Query band taking care of aliases.

        :raises: :py:class:`KeyError`
        """
        if isinstance(band, str):
            band = self.collection.band_key(band)
        return self.bands[band]

    def __len__(self) -> int:
        return len(self.bands)

    def __iter__(self) -> Iterator[BandKey]:
        yield from self.bands

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
        return replace(self, bands={k: band.strip() for k, band in self.bands.items()})

    def assets(self) -> Dict[str, List[RasterSource]]:
        """
        Extract bands grouped by asset they belong to.
        """
        assets: Dict[str, List[Tuple[int, RasterSource]]] = {}
        for (asset, idx), src in self.bands.items():
            assets.setdefault(asset, []).append((idx, src))

        return {
            k: [src for _, src in sorted(srcs, key=lambda x: x[0])]
            for k, srcs in assets.items()
        }

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


@dataclass(frozen=True)
class MDParseConfig:
    """Item parsing config."""

    band_defaults: RasterBandMetadata = field(default_factory=RasterBandMetadata)
    band_cfg: Dict[str, RasterBandMetadata] = field(default_factory=dict)
    aliases: Dict[str, BandKey] = field(default_factory=dict)
    ignore_proj: bool = False

    @staticmethod
    def from_dict(collection_id: str, cfg=Dict[str, Any]) -> "MDParseConfig":
        _cfg = copy(cfg.get("*", {}))
        _cfg.update(cfg.get(collection_id, {}))
        band_defaults, band_cfg = _norm_band_cfg(_cfg.get("assets", {}))

        aliases = {
            alias: ((band, 1) if isinstance(band, str) else band)
            for alias, band in _cfg.get("aliases", {}).items()
        }
        ignore_proj: bool = _cfg.get("ignore_proj", False)
        return MDParseConfig(
            band_defaults=band_defaults,
            band_cfg=band_cfg,
            ignore_proj=ignore_proj,
            aliases=aliases,
        )


def _norm_band_cfg(
    cfg: Dict[str, Any]
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
