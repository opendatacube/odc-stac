"""Metadata and data loading model classes."""

import datetime as dt
from dataclasses import astuple, dataclass, replace
from typing import (
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from odc.geo import CRS, Geometry
from odc.geo.geobox import GeoBox

T = TypeVar("T")


@dataclass(eq=True, frozen=True)
class RasterBandMetadata:
    """
    Common raster metadata per band.

    We assume that all assets of the same name have the same "structure" across different items
    within a collection. Specifically, that they encode pixels with the same data type, use the same
    ``nodata`` value and have common units.

    These values are extracted from the ``eo:bands`` extension, but can also be supplied by the user
    from the config.
    """

    data_type: Optional[str] = None
    """Numpy compatible dtype string."""

    nodata: Optional[float] = None
    """Nodata marker/fill_value."""

    unit: str = "1"
    """Units of the pixel data."""

    def __dask_tokenize__(self):
        return astuple(self)


@dataclass(eq=True, frozen=True)
class RasterCollectionMetadata(Mapping[str, RasterBandMetadata]):
    """
    Information about raster data in a collection.

    We assume that assets with the same names have the same kind of raster data across items within
    a collection. This is built from the combination of data collected from STAC and user
    configuration if supplied.
    """

    name: str
    """Collection name."""

    bands: Dict[str, RasterBandMetadata]
    """
    Bands are assets that contain raster data.

    This controls which assets are extracted from STAC.
    """

    aliases: Dict[str, str]
    """
    Alias map ``alias -> asset name``.

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

    def band_aliases(self) -> Dict[str, List[str]]:
        """
        Compute inverse of alias mapping.

        :return:
          Mapping from canonical name to a list of defined aliases.
        """
        out: Dict[str, List[str]] = {}
        for alias, cn in self.aliases.items():
            out.setdefault(cn, []).append(alias)
        return out

    def resolve_bands(
        self, bands: Optional[Union[str, Sequence[str]]] = None
    ) -> Dict[str, RasterBandMetadata]:
        """
        Query bands taking care of aliases.
        """
        return _resolve_aliases(self.bands, self.aliases, bands)

    def canonical_name(self, band: str) -> str:
        """
        Canonical name for an alias.
        """
        return self.aliases.get(band, band)

    def __getitem__(self, band: str) -> RasterBandMetadata:
        """
        Query band taking care of aliases.

        :raises: :py:class:`KeyError`
        """
        return self.bands[self.canonical_name(band)]

    def __len__(self) -> int:
        return len(self.bands)

    def __iter__(self) -> Iterator[str]:
        yield from self.bands

    def __contains__(self, __o: object) -> bool:
        return __o in self.bands or __o in self.aliases

    def __dask_tokenize__(self):
        return astuple(self)


@dataclass(eq=True, frozen=True)
class RasterSource:
    """
    Captures known information about a single band.
    """

    uri: str
    """Asset location."""

    band: int = 1
    """One based band index (default=1)."""

    subdataset: Optional[str] = None
    """Used for netcdf/hdf5 sources."""

    geobox: Optional[GeoBox] = None
    """Data footprint/shape/projection if known."""

    meta: Optional[RasterBandMetadata] = None
    """Expected raster dtype/nodata."""

    def strip(self) -> "RasterSource":
        """
        Copy with minimal data only.

        Removes geobox and meta info as they are not needed for data loading.
        """
        return RasterSource(self.uri, self.band, self.subdataset)

    def __dask_tokenize__(self):
        return (self.uri, self.band, self.subdataset)


@dataclass(eq=True, frozen=True)
class ParsedItem(Mapping[str, RasterSource]):
    """
    Captures essentials parts for data loading from a STAC Item.

    Only includes raster bands of interest.
    """

    id: str
    """Item id copied from STAC."""

    collection: RasterCollectionMetadata
    """Collection this Item is part of."""

    bands: Dict[str, RasterSource]
    """Raster bands."""

    geometry: Optional[Geometry] = None
    """Footprint of the dataset."""

    datetime: Optional[dt.datetime] = None
    """Nominal timestamp."""

    datetime_range: Tuple[Optional[dt.datetime], Optional[dt.datetime]] = None, None
    """Time period covered."""

    href: Optional[str] = None
    """Self link from stac item."""

    def geoboxes(self, bands: Optional[Sequence[str]] = None) -> Tuple[GeoBox, ...]:
        """
        Unique ``GeoBox``s, highest resolution first.

        :param bands: which bands to consider, default is all
        """
        if bands is None:
            bands = list(self.bands)

        def _resolution(g: GeoBox) -> float:
            return min(g.resolution.map(abs).xy)  # type: ignore

        gbx: Set[GeoBox] = set()
        aliases = self.collection.aliases
        for name in bands:
            b = self.bands.get(aliases.get(name, name), None)
            if b is not None:
                if b.geobox is not None:
                    gbx.add(b.geobox)

        return tuple(sorted(gbx, key=_resolution))

    def crs(self, bands: Optional[Sequence[str]] = None) -> Optional[CRS]:
        """
        First non-null CRS across assets.
        """
        for gbox in self.geoboxes(bands):
            if gbox.crs is not None:
                return gbox.crs

        return None

    def resolve_bands(
        self, bands: Optional[Union[str, Sequence[str]]] = None
    ) -> Dict[str, RasterSource]:
        """
        Query bands taking care of aliases.
        """
        return _resolve_aliases(self.bands, self.collection.aliases, bands)

    def __getitem__(self, band: str) -> RasterSource:
        """
        Query band taking care of aliases.

        :raises: :py:class:`KeyError`
        """
        return self.bands[self.collection.canonical_name(band)]

    def __len__(self) -> int:
        return len(self.bands)

    def __iter__(self) -> Iterator[str]:
        yield from self.bands

    def __contains__(self, k: object) -> bool:
        if not isinstance(k, str):
            return False
        return self.collection.canonical_name(k) in self.bands

    @property
    def nominal_datetime(self) -> dt.datetime:
        """
        Resolve timestamp to a single value.

        - datetime if set
        - start_datetime if set
        - end_datetime if set
        - raise ValueError otherwise
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


@dataclass
class RasterLoadParams:
    """
    Captures data loading configuration.
    """

    dtype: Optional[str] = None
    """Output dtype, default same as source."""

    fill_value: Optional[float] = None
    """Value to use for missing pixels."""

    src_nodata_fallback: Optional[float] = None
    """
    Fallback ``nodata`` marker for source.

    Used to deal with broken data sources. If file is missing ``nodata`` marker and
    ``src_nodata_fallback`` is set then treat source pixels with that value as missing.
    """

    src_nodata_override: Optional[float] = None
    """
    Override ``nodata`` marker for source.

    Used to deal with broken data sources. Ignore ``nodata`` marker of the source file even if
    present and use this value instead.
    """

    use_overviews: bool = True
    """
    Disable use of overview images.

    Set to ``False`` to always read from the main image ignoring overview images
    even when present in the data source.
    """

    resampling: str = "nearest"
    """Resampling method to use."""

    @staticmethod
    def same_as(src: Union[RasterBandMetadata, RasterSource]) -> "RasterLoadParams":
        """Construct from source object."""
        if isinstance(src, RasterBandMetadata):
            meta = src
        else:
            meta = src.meta or RasterBandMetadata()

        dtype = meta.data_type
        if dtype is None:
            dtype = "float32"

        return RasterLoadParams(dtype=dtype, fill_value=meta.nodata)

    @property
    def nearest(self) -> bool:
        """Report True if nearest resampling is used."""
        return self.resampling == "nearest"

    def __dask_tokenize__(self):
        return astuple(self)


def _resolve_aliases(
    src: Mapping[str, T],
    aliases: Mapping[str, str],
    bands: Optional[Union[str, Sequence[str]]] = None,
) -> Dict[str, T]:
    if bands is None:
        bands = list(src)
    if isinstance(bands, str):
        bands = [bands]

    out: Dict[str, T] = {}
    for name in bands:
        src_name = aliases.get(name, name)
        if src_name not in src:
            raise ValueError(f"No such band or alias: '{name}'")
        out[name] = src[src_name]

    return out


def _convert_to_solar_time(utc: dt.datetime, longitude: float) -> dt.datetime:
    # offset_seconds snapped to 1 hour increments
    #    1/15 == 24/360 (hours per degree of longitude)
    offset_seconds = int(longitude / 15) * 3600
    return utc + dt.timedelta(seconds=offset_seconds)
