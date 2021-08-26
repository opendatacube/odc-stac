""" Indexing related helper methods.
"""
from ..stac._version import __version__

from ._index import (
    from_metadata_stream,
    from_yaml_doc_stream,
    parse_doc_stream,
    bin_dataset_stream,
    bin_dataset_stream2,
    dataset_count,
    count_by_year,
    count_by_month,
    chop_query_by_time,
    time_range,
    month_range,
    season_range,
    ordered_dss,
    chopped_dss,
    all_datasets,
    product_from_yaml,
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

from ._yaml import (
    render_eo3_yaml,
)

from ._grouper import (
    group_by_nothing,
    solar_offset,
)

__all__ = (
    "from_yaml_doc_stream",
    "from_metadata_stream",
    "parse_doc_stream",
    "bin_dataset_stream",
    "bin_dataset_stream2",
    "dataset_count",
    "count_by_year",
    "count_by_month",
    "chop_query_by_time",
    "time_range",
    "month_range",
    "season_range",
    "ordered_dss",
    "chopped_dss",
    "all_datasets",
    "product_from_yaml",
    "odc_uuid",
    "utm_region_code",
    "utm_zone_to_epsg",
    "utm_tile_dss",
    "mk_utm_gs",
    "render_eo3_yaml",
    "group_by_nothing",
    "solar_offset",
)
