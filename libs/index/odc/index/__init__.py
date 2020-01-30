""" Indexing related helper methods.
"""

from . _index import (
    from_metadata_stream,
    from_yaml_doc_stream,
    bin_dataset_stream,
    dataset_count,
    count_by_year,
    count_by_month,
    chop_query_by_time,
    time_range,
    month_range,
    season_range,
    ordered_dss,
)

from ._eo3 import (
    eo3_lonlat_bbox,
    eo3_grid_spatial,
)

from ._uuid import (
    odc_uuid,
)

from ._utm import (
    utm_region_code,
    utm_zone_to_epsg,
    utm_tile_dss,
    mk_utm_gs,
)

__all__ = (
    "from_yaml_doc_stream",
    "from_metadata_stream",
    "bin_dataset_stream",
    "dataset_count",
    "count_by_year",
    "count_by_month",
    "chop_query_by_time",
    "time_range",
    "month_range",
    "season_range",
    "ordered_dss",
    "eo3_lonlat_bbox",
    "eo3_grid_spatial",
    "odc_uuid",
    "utm_region_code",
    "utm_zone_to_epsg",
    "utm_tile_dss",
    "mk_utm_gs",
)
