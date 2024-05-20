"""Metadata and data loading model classes."""

from __future__ import annotations

from dataclasses import astuple, dataclass, field
from typing import (
    Any,
    ContextManager,
    Dict,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
from odc.geo.geobox import GeoBox
from odc.geo.roi import NormalizedROI

T = TypeVar("T")

BandKey = Tuple[str, int]
"""Asset Name, band index within an asset (1 based)."""

BandIdentifier = Union[str, BandKey]
"""Alias or canonical band identifier."""

BandQuery = Optional[Union[str, Sequence[str]]]
"""One|All|Some bands"""


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

    dims: Tuple[str, ...] = ()
    """Dimension names for this band.

    e.g. ("y", "x", "wavelength")
    """

    def with_defaults(self, defaults: "RasterBandMetadata") -> "RasterBandMetadata":
        """
        Merge with another metadata object, using self as the primary source.

        If a field is None in self, use the value from defaults.
        """
        return RasterBandMetadata(
            data_type=with_default(self.data_type, defaults.data_type),
            nodata=with_default(self.nodata, defaults.nodata),
            unit=with_default(self.unit, defaults.unit),
            dims=self.dims or defaults.dims,
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


@dataclass(eq=True)
class FixedCoord:
    """
    Encodes extra coordinate info.
    """

    name: str
    values: Sequence[Any]
    dtype: str = ""
    dim: str = ""
    units: str = "1"

    def __post_init__(self):
        if not self.dtype:
            self.dtype = np.array(self.values).dtype.str
        if not self.dim:
            self.dim = self.name

    def _repr_json_(self) -> Dict[str, Any]:
        """
        Return a JSON serializable representation of the FixedCoord object.
        """
        return {
            "name": self.name,
            "values": list(self.values),
            "dim": self.dim,
            "dtype": self.dtype,
            "units": self.units,
        }


@dataclass(eq=True, frozen=True)
class RasterGroupMetadata:
    """
    STAC Collection/Datacube Product abstraction.
    """

    bands: Dict[BandKey, RasterBandMetadata]
    """
    Bands are assets that contain raster data.

    This controls which assets are extracted from STAC.
    """

    aliases: dict[str, list[BandKey]] = field(default_factory=dict)
    """
    Alias map ``alias -> [(asset, idx),...]``.

    Used to rename bands at load time.
    """

    extra_dims: dict[str, int] = field(default_factory=dict)
    """
    Expected extra dimensions other than time and spatial.

    Must be same size across items/datasets.
    """

    extra_coords: Sequence[FixedCoord] = ()
    """
    Coordinates for extra dimensions.

    Must be same values across items/datasets.
    """

    def _repr_json_(self) -> Dict[str, Any]:
        """
        Return a JSON serializable representation of the RasterGroupMetadata object.
        """
        # pylint: disable=protected-access
        return {
            "bands": {
                f"{name}.{idx}": v._repr_json_()
                for (name, idx), v in self.bands.items()
            },
            "aliases": self.aliases,
            "extra_dims": self.extra_dims,
            "extra_coords": [c._repr_json_() for c in self.extra_coords],
        }

    def extra_dims_full(self) -> Dict[str, int]:
        dims = {**self.extra_dims}
        for coord in self.extra_coords:
            if coord.dim not in dims:
                dims[coord.dim] = len(coord.values)

        return dims


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


MultiBandRasterSource = Union[
    Mapping[str, RasterSource],
    Mapping[BandIdentifier, RasterSource],
]
"""Mapping from band name to RasterSource."""


@dataclass
class RasterLoadParams:
    """
    Captures data loading configuration.
    """

    # pylint: disable=too-many-instance-attributes

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

    dims: Tuple[str, ...] = ()
    """Dimension names for this band."""

    @property
    def ydim(self) -> int:
        if self.dims:
            return self.dims.index("y")
        return 0

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

        return RasterLoadParams(dtype=dtype, fill_value=meta.nodata, dims=meta.dims)

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
            "dims": self.dims,
        }


class MDParser(Protocol):
    """
    Protocol for metadata parsers.

    - Parse group level metadata
      - data bands andn their expected type
      - extra dimensions and coordinates
    - Extract driver specific data
    """

    def extract(self, md: Any) -> RasterGroupMetadata: ...
    def driver_data(self, md: Any, band_key: BandKey) -> Any: ...


class RasterReader(Protocol):
    """
    Protocol for raster readers.
    """

    # pylint: disable=too-few-public-methods

    def read(
        self,
        cfg: RasterLoadParams,
        dst_geobox: GeoBox,
        dst: Optional[np.ndarray] = None,
    ) -> Tuple[NormalizedROI, np.ndarray]: ...


class ReaderDriver(Protocol):
    """
    Protocol for reader drivers.
    """

    def new_load(self, chunks: None | Dict[str, int] = None) -> Any: ...

    def finalise_load(self, load_state: Any) -> Any: ...

    def capture_env(self) -> Dict[str, Any]: ...

    def restore_env(
        self, env: Dict[str, Any], load_state: Any
    ) -> ContextManager[Any]: ...

    def open(self, src: RasterSource, ctx: Any) -> RasterReader: ...

    @property
    def md_parser(self) -> MDParser | None: ...


ReaderDriverSpec = Union[str, ReaderDriver]


def is_reader_driver(x: Any) -> bool:
    """
    Check if x is a ReaderDriver.
    """
    expected_attributes = [x for x in dir(ReaderDriver) if not x.startswith("_")]
    return all(hasattr(x, a) for a in expected_attributes)


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


def norm_key(k: BandIdentifier) -> BandKey:
    """
    ("band", i) -> ("band", i)
    "band" -> ("band", 1)
    "band.3" -> ("band", 3)
    """
    if isinstance(k, str):
        parts = k.rsplit(".", 1)
        if len(parts) == 2:
            return parts[0], int(parts[1])
        return (k, 1)
    return k
