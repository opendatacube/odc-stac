"""Metadata and data loading model classes."""

from dataclasses import astuple, dataclass
from typing import Any, ContextManager, Dict, Optional, Protocol, Tuple, Union

import numpy as np
from odc.geo.geobox import GeoBox
from odc.geo.roi import NormalizedROI


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

    dims: Optional[Tuple[str, ...]] = None
    """Dimension names for this band."""

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

    fail_on_error: bool = True
    """Quit on the first error or continue."""

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


class SomeReader(Protocol):
    """
    Protocol for readers.
    """

    def capture_env(self) -> Dict[str, Any]: ...

    def restore_env(self, env: Dict[str, Any]) -> ContextManager[Any]: ...

    def read(
        self,
        src: RasterSource,
        cfg: RasterLoadParams,
        dst_geobox: GeoBox,
        dst: Optional[np.ndarray] = None,
    ) -> Tuple[NormalizedROI, np.ndarray]: ...


BAND_DEFAULTS = RasterBandMetadata("float32", None, "1")


def norm_band_metadata(
    v: Union[RasterBandMetadata, Dict[str, Any]],
    fallback: RasterBandMetadata = BAND_DEFAULTS,
) -> RasterBandMetadata:
    if isinstance(v, RasterBandMetadata):
        return v
    return RasterBandMetadata(
        v.get("data_type", fallback.data_type),
        v.get("nodata", fallback.nodata),
        v.get("unit", fallback.unit),
    )
