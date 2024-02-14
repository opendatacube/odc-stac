"""Metadata and data loading model classes."""

from __future__ import annotations

from dataclasses import astuple, dataclass
from typing import (
    Any,
    ContextManager,
    Dict,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
from odc.geo.geobox import GeoBox
from odc.geo.roi import NormalizedROI

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

    dims: Optional[Tuple[str, ...]] = None
    """Dimension names for this band."""

    def with_defaults(self, defaults: "RasterBandMetadata") -> "RasterBandMetadata":
        """
        Merge with another metadata object, using self as the primary source.

        If a field is None in self, use the value from defaults.
        """
        return RasterBandMetadata(
            data_type=with_default(self.data_type, defaults.data_type),
            nodata=with_default(self.nodata, defaults.nodata),
            unit=with_default(self.unit, defaults.unit),
            dims=with_default(self.dims, defaults.dims),
        )

    def __dask_tokenize__(self):
        return astuple(self)

    def _repr_json_(self) -> Dict[str, Any]:
        """
        Return a JSON serializable representation of the RasterBandMetadata object.
        """
        return {
            "data_type": self.data_type,
            "nodata": self.nodata,
            "unit": self.unit,
            "dims": self.dims,
        }


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

    driver_data: Any = None
    """IO Driver specific extra data."""

    def strip(self) -> "RasterSource":
        """
        Copy with minimal data only.

        Removes geobox, as it's not needed for data loading.
        """
        return RasterSource(
            self.uri,
            band=self.band,
            subdataset=self.subdataset,
            geobox=None,
            meta=self.meta,
            driver_data=self.driver_data,
        )

    def __dask_tokenize__(self):
        return (self.uri, self.band, self.subdataset)

    def _repr_json_(self) -> Dict[str, Any]:
        """
        Return a JSON serializable representation of the RasterSource object.
        """
        doc = {
            "uri": self.uri,
            "band": self.band,
        }

        if self.subdataset is not None:
            doc["subdataset"] = self.subdataset

        if self.meta is not None:
            doc.update(self.meta._repr_json_())  # pylint: disable=protected-access

        gbox = self.geobox
        if gbox is not None:
            doc["crs"] = str(gbox.crs)
            doc["transform"] = [*gbox.transform][:6]
            doc["shape"] = gbox.shape.yx

        return doc


MultiBandRasterSource = Mapping[Union[str, Tuple[str, int]], RasterSource]
"""Mapping from band name to RasterSource."""


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

    def _repr_json_(self) -> Dict[str, Any]:
        """
        Return a JSON serializable representation of the RasterLoadParams object.
        """
        return {
            "dtype": self.dtype,
            "fill_value": self.fill_value,
            "src_nodata_fallback": self.src_nodata_fallback,
            "src_nodata_override": self.src_nodata_override,
            "use_overviews": self.use_overviews,
            "resampling": self.resampling,
            "fail_on_error": self.fail_on_error,
        }


class MDParser(Protocol):
    """
    Protocol for metadata parsers.

    - Compute bands from an asset
    - Compute band aliases from an asset
    - Extract driver specific data
    """

    def bands(self, md: Any, name: str) -> Tuple[RasterBandMetadata, ...]: ...
    def aliases(self, md: Any, name: str) -> Tuple[str, ...]: ...
    def driver_data(self, md: Any, name: str, idx: int) -> Any: ...


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

    md_parser: MDParser | None


BAND_DEFAULTS = RasterBandMetadata("float32", None, "1")


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


def norm_nodata(nodata) -> Union[float, None]:
    if nodata is None:
        return None
    if isinstance(nodata, (int, float)):
        return nodata
    return float(nodata)


def norm_band_metadata(
    v: Union[RasterBandMetadata, Dict[str, Any]],
    fallback: RasterBandMetadata = BAND_DEFAULTS,
) -> RasterBandMetadata:
    if isinstance(v, RasterBandMetadata):
        return v
    return RasterBandMetadata(
        v.get("data_type", fallback.data_type),
        v.get("nodata", norm_nodata(fallback.nodata)),
        v.get("unit", fallback.unit),
    )
