"""Metadata and data loading model classes."""

from dataclasses import dataclass
from typing import Dict, Optional

from odc.geo.geobox import GeoBox


@dataclass
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


@dataclass
class RasterCollectionMetadata:
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


@dataclass
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


@dataclass
class ParsedItem:
    """
    Captures essentials parts for data loading from a STAC Item.

    Only includes raster bands of interest.
    """

    collection: RasterCollectionMetadata
    """Collection this Item is part of."""

    bands: Dict[str, RasterSource]
    """Raster bands."""


@dataclass
class RasterLoadParams:
    """
    Captures data loading configuration.
    """

    src: RasterSource
    """Source band."""

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

    def __post_init__(self) -> None:
        meta = self.src.meta

        if meta is None:
            return

        if self.dtype is None:
            self.dtype = meta.data_type

        if self.fill_value is None:
            self.fill_value = meta.nodata
