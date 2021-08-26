from datetime import timedelta
from typing import List, Dict, Any, Optional, Iterator, Hashable, Iterable
import xarray as xr
import pandas as pd
import numpy as np
from datacube.model import Dataset
from datacube.utils.dates import normalise_dt
from datacube.utils.geometry import Geometry


def mid_longitude(geom: Geometry) -> float:
    ((lon,), _) = geom.centroid.to_crs("epsg:4326").xy
    return lon


def solar_offset(geom: Geometry, precision: str = "h") -> timedelta:
    """
    Given a geometry compute offset to add to UTC timestamp to get solar day right.

    This only work when geometry is "local enough".
    :param precision: one of ``'h'`` or ``'s'``, defaults to hour precision
    """
    lon = mid_longitude(geom)

    if precision == "h":
        return timedelta(hours=int(lon * 24 / 360 + 0.5))

    # 240 == (24*60*60)/360 (seconds of a day per degree of longitude)
    return timedelta(seconds=int(lon * 240))


def key2num(
    objs: Iterable[Hashable], reverse_map: Optional[Dict[int, Any]] = None
) -> Iterator[int]:
    """
    Given a sequence of hashable objects return sequence of numeric ids starting from 0.
    For example ``'A' 'B' 'A' 'A' 'C' -> 0 1 0 0 2``
    """
    o2id: Dict[Any, int] = {}
    c = 0
    for obj in objs:
        _c = o2id.setdefault(obj, c)
        if _c == c:
            c = c + 1
            if reverse_map is not None:
                reverse_map[_c] = obj
        yield _c


def group_by_nothing(
    dss: List[Dataset], solar_day_offset: Optional[timedelta] = None
) -> xr.DataArray:
    """
    Construct "sources" just like ``.group_dataset`` but with every slice
    containing just one Dataset object wrapped in a tuple.

    Time -> (Dataset,)
    """
    dss = sorted(dss, key=lambda ds: (normalise_dt(ds.center_time), ds.id))
    time = [normalise_dt(ds.center_time) for ds in dss]
    solar_day = None

    if solar_day_offset is not None:
        solar_day = np.asarray(
            [(dt + solar_day_offset).date() for dt in time], dtype="datetime64[D]"
        )

    idx = np.arange(0, len(dss), dtype="uint32")
    uuids = np.empty(len(dss), dtype="O")
    data = np.empty(len(dss), dtype="O")
    grid2crs: Dict[int, Any] = {}
    grid = list(key2num((ds.crs for ds in dss), grid2crs))

    for i, ds in enumerate(dss):
        data[i] = (ds,)
        uuids[i] = ds.id

    coords = [np.asarray(time, dtype="datetime64[ms]"), idx, uuids, grid]
    names = ["time", "idx", "uuid", "grid"]
    if solar_day is not None:
        coords.append(solar_day)
        names.append("solar_day")

    coord = pd.MultiIndex.from_arrays(coords, names=names)

    return xr.DataArray(
        data=data, coords=dict(spec=coord), attrs={"grid2crs": grid2crs}, dims=("spec",)
    )
