"""These should probably be in datacube library."""
import datetime
import json
import sys
from random import randint
from types import SimpleNamespace
from typing import Callable, Iterable, Iterator, Optional, Set, Tuple
from uuid import UUID
from warnings import warn

import psycopg2
from datacube import Datacube
from datacube.api.query import Query
from datacube.index.hl import Doc2Dataset
from datacube.index.index import default_metadata_type_docs
from datacube.model import Dataset, DatasetType, Range, metadata_from_doc
from datacube.storage import measurement_paths
from datacube.utils.documents import load_documents, parse_yaml
from pandas import Period

from ._grouper import solar_offset


def from_metadata_stream(metadata_stream, index, **kwargs):
    """
    Raw metadata to Dataset stream converter.

    Given a stream of (uri, metadata) tuples convert them into Datasets, using
    supplied index and options for Doc2Dataset.


    **kwargs**:
    - skip_lineage
    - verify_lineage
    - fail_on_missing_lineage
    - products
    - exclude_products

    returns a sequence of tuples where each tuple is either

        (Dataset, None) or (None, error_message)
    """
    doc2ds = Doc2Dataset(index, **kwargs)

    for uri, metadata in metadata_stream:
        if metadata is None:
            yield (None, f"Error: empty doc {uri}")
        else:
            ds, err = doc2ds(metadata, uri)
            if ds is not None:
                yield (ds, None)
            else:
                yield (None, f"Error: {uri}, {err}")


def parse_doc_stream(doc_stream, on_error=None, transform=None):
    """
    Replace doc bytes/strings with parsed dicts.

       Stream[(uri, bytes)] -> Stream[(uri, dict)]


    :param doc_stream: sequence of (uri, doc: bytes|string)
    :param on_error: Callback uri, doc, exception -> None
    :param transform: dict -> dict if supplied also apply further transform on parsed document

    On output doc is replaced with python dict parsed from yaml, or with None
    if parsing/transform error occurred.
    """
    for uri, doc in doc_stream:
        try:
            if uri.endswith(".json"):
                metadata = json.loads(doc)
            else:
                metadata = parse_yaml(doc)

            if transform is not None:
                metadata = transform(metadata)
        except Exception as e:  # pylint: disable=broad-except
            if on_error is not None:
                on_error(uri, doc, e)
            metadata = None

        yield uri, metadata


def from_yaml_doc_stream(doc_stream, index, logger=None, transform=None, **kwargs):
    """
    Stream of yaml documents to a stream of Dataset results.

    Stream[(path, bytes|str)] -> Stream[(Dataset, None)|(None, error_message)]

    :param doc_stream: sequence of (uri, doc: byges|string)
    :param on_error: Callback uri, doc, exception -> None
    :param logger:  Logger object for printing errors or None
    :param transform: dict -> dict if supplied also apply further transform on parsed document
    :param kwargs: passed on to from_metadata_stream

    """

    def on_parse_error(uri, doc, err):
        # pylint: disable=unused-argument
        if logger is not None:
            logger.error(f"Failed to parse: {uri}")
        else:
            print(f"Failed to parse: {uri}", file=sys.stderr)

    metadata_stream = parse_doc_stream(
        doc_stream, on_error=on_parse_error, transform=transform
    )
    return from_metadata_stream(metadata_stream, index, **kwargs)


def dataset_count(index, **query):
    """Return number of datasets matching a query."""
    return index.datasets.count(**Query(**query).search_terms)


def count_by_year(index, product, min_year=None, max_year=None):
    """
    Return dictionary Int->Int: `year` -> `dataset count for this year`.

    Only non-empty years are reported.
    """
    # TODO: get min/max from datacube properly
    if min_year is None:
        min_year = 1970
    if max_year is None:
        max_year = datetime.datetime.now().year

    year_count = (
        (year, dataset_count(index, product=product, time=str(year)))
        for year in range(min_year, max_year + 1)
    )

    return {year: c for year, c in year_count if c > 0}


def count_by_month(index, product, year):
    """
    Return 12 integer tuple.

    counts for January, February ... December
    """
    return tuple(
        dataset_count(index, product=product, time=f"{year}-{month:02d}")
        for month in range(1, 12 + 1)
    )


def time_range(begin, end, freq="m"):
    """
    Return tuples of datetime objects aligned to boundaries of requested period.

    (month is default).
    """
    tzinfo = begin.tzinfo
    t = Period(begin, freq)

    def to_pydate(t):
        return t.to_pydatetime(warn=False).replace(tzinfo=tzinfo)

    while True:
        t0, t1 = map(to_pydate, (t.start_time, t.end_time))
        if t0 > end:
            break

        yield (max(t0, begin), min(t1, end))
        t += 1


def month_range(
    year: int, month: int, n: int
) -> Tuple[datetime.datetime, datetime.datetime]:
    """
    Construct month aligned time range.

    Return time range covering n months starting from year, month
    month 1..12
    month can also be negative
    2020, -1 === 2019, 12
    """
    if month < 0:
        return month_range(year - 1, 12 + month + 1, n)

    y2 = year
    m2 = month + n
    if m2 > 12:
        m2 -= 12
        y2 += 1
    dt_eps = datetime.timedelta(microseconds=1)

    return (
        datetime.datetime(year=year, month=month, day=1),
        datetime.datetime(year=y2, month=m2, day=1) - dt_eps,
    )


def season_range(year: int, season: str) -> Tuple[datetime.datetime, datetime.datetime]:
    """
    Season is one of djf, mam, jja, son.

    DJF for year X starts in Dec X-1 and ends in Feb X.
    """
    seasons = dict(djf=-1, mam=3, jja=6, son=9)

    start_month = seasons.get(season.lower())
    if start_month is None:
        raise ValueError(f"No such season {season}, valid seasons are: djf,mam,jja,son")
    return month_range(year, start_month, 3)


def chop_query_by_time(q: Query, freq: str = "m") -> Iterator[Query]:
    """
    Split query along time dimension.

    Given a query over longer period of time, chop it up along the time dimension
    into smaller queries each covering a shorter time period (year, month, week or day).
    """
    qq = dict(**q.search_terms)
    time = qq.pop("time", None)
    if time is None:
        raise ValueError("Need time range in the query")

    for (t0, t1) in time_range(time.begin, time.end, freq=freq):
        yield Query(**qq, time=Range(t0, t1))


def ordered_dss(dc: Datacube, freq: str = "m", key=None, **query):
    """
    Emulate "order by time" streaming interface for datacube queries.

        Basic idea is to perform a lot of smaller queries (shorter time
        periods), sort results then yield them to the calling code.

    :param dc: Datacube instance

    :param freq: 'm' month sized chunks, 'w' week sized chunks, 'd' day

    :param key: Optional sorting function Dataset -> Comparable, for example
                ``lambda ds: (ds.center_time, ds.metadata.region_code)``
    """
    qq = Query(**query)
    if key is None:
        key = lambda ds: ds.center_time

    _last_uuids: Set[UUID] = set()

    for q in chop_query_by_time(qq, freq=freq):
        _dss = dc.find_datasets(**q.search_terms)
        dss = [ds for ds in _dss if ds.id not in _last_uuids]
        _last_uuids = {ds.id for ds in _dss}
        dss.sort(key=key)
        yield from dss


def chopped_dss(dc: Datacube, freq: str = "m", **query):
    """
    Emulate streaming interface for datacube queries.

    Basic idea is to perform a lot of smaller queries (shorter time
    periods)
    """
    qq = Query(**query)
    _last_uuids: Set[UUID] = set()

    for q in chop_query_by_time(qq, freq=freq):
        _dss = dc.find_datasets(**q.search_terms)
        dss = [ds for ds in _dss if ds.id not in _last_uuids]
        _last_uuids = {ds.id for ds in _dss}
        yield from dss


def bin_dataset_stream(gridspec, dss, cells, persist=None):
    """
    Intersect Grid Spec cells with Datasets.

    :param gridspec: GridSpec
    :param dss: Sequence of datasets (can be lazy)
    :param cells: Dictionary to populate with tiles
    :param persist: Dataset -> SomeThing mapping, defaults to keeping dataset id only

    The ``cells`` dictionary is a mapping from (x,y) tile index to object with the
    following properties:

     .idx     - tile index (x,y)
     .geobox  - tile geobox
     .utc_offset - timedelta to add to timestamp to get day component in local time
     .dss     - list of UUIDs, or results of ``persist(dataset)`` if custom ``persist`` is supplied
    """
    geobox_cache = {}

    def default_persist(ds):
        return ds.id

    def register(tile, geobox, val):
        cell = cells.get(tile)
        if cell is None:
            utc_ofset = solar_offset(geobox.extent)
            cells[tile] = SimpleNamespace(
                geobox=geobox, idx=tile, utc_offset=utc_ofset, dss=[val]
            )
        else:
            cell.dss.append(val)

    if persist is None:
        persist = default_persist

    for ds in dss:
        ds_val = persist(ds)

        if ds.extent is None:
            warn(f"Dataset without extent info: {str(ds.id)}")
            continue

        for tile, geobox in gridspec.tiles_from_geopolygon(
            ds.extent, geobox_cache=geobox_cache
        ):
            register(tile, geobox, ds_val)

        yield ds


def bin_dataset_stream2(gridspec, dss, geobox_cache=None):
    """
    For every input dataset compute tiles of the GridSpec it overlaps with.

    Iterable[Dataset] -> Iterator[(Dataset, List[Tuple[int, int]])]
    """
    if geobox_cache is None:
        geobox_cache = {}

    for ds in dss:
        if ds.extent is None:
            warn(f"Dataset without extent info: {ds.id}")
            tiles = []
        else:
            tiles = [
                tile
                for tile, _ in gridspec.tiles_from_geopolygon(
                    ds.extent, geobox_cache=geobox_cache
                )
            ]

        yield ds, tiles


def all_datasets(
    dc: Datacube, product: str, read_chunk: int = 1000, limit: Optional[int] = None
):
    """
    Properly lazy version of ``dc.find_datasets_lazy(product=product)``.

    Uses db cursors to reduce latency of results arriving from the database, original
    method in datacube fetches entire query result from the database and only then lazily
    constructs Dataset objects from the returned table.

    :param dc: Datacube object
    :param product: Product name to extract
    :param read_chunk: Number of datasets per page
    :param limit: Optionally constrain query to just return that many dataset
    """
    assert isinstance(limit, (int, type(None)))

    db = psycopg2.connect(str(dc.index.url))
    _limit = "" if limit is None else f"LIMIT {limit}"

    _product = dc.index.products.get_by_name(product)
    if _product is None:
        raise ValueError(f"No such product: {product}")

    query = f"""select
jsonb_build_object(
  'product', %(product)s,
  'uris', array((select _loc_.uri_scheme ||':'||_loc_.uri_body
                 from agdc.dataset_location as _loc_
                 where _loc_.dataset_ref = agdc.dataset.id and _loc_.archived is null
                 order by _loc_.added desc, _loc_.id desc)),
  'metadata', metadata) as dataset
from agdc.dataset
where archived is null
and dataset_type_ref = (select id from agdc.dataset_type where name = %(product)s)
{_limit};
"""
    cursor_name = f"c{randint(0, 0xFFFF):04X}"
    with db.cursor(name=cursor_name) as cursor:
        cursor.execute(query, dict(product=product))

        while True:
            chunk = cursor.fetchmany(read_chunk)
            if not chunk:
                break
            for (ds,) in chunk:
                yield Dataset(_product, ds["metadata"], ds["uris"])


def product_from_yaml(path: str, dc: Optional[Datacube] = None) -> DatasetType:
    """
    Make product definition from yaml file without access to the database.

    .. note:

         access to database is only needed when non-standard metadata types are used.

    :param path: File path or a URL pointing to yaml definition of the product
    :param dc: Optional datacube instance (used to query MetadataType, only
               used if non-standard metadata type is used in the product)
    """
    standard_metadata_types = {
        d["name"]: metadata_from_doc(d) for d in default_metadata_type_docs()
    }

    product, *_ = load_documents(path)

    metadata = standard_metadata_types.get(product.get("metadata_type"), None)
    if metadata is not None:
        # Standard metadata we can do this without DB
        return DatasetType(metadata, product)

    # Not eo|eo3 see if we can get that from DB
    if dc is None:
        dc = Datacube()

    return dc.index.products.from_doc(product)


def patch_urls(
    ds: Dataset, edit: Callable[[str], str], bands: Optional[Iterable[str]] = None
) -> Dataset:
    """
    Map function over dataset measurement urls.

    :param ds: Dataset to edit in place
    :param edit: Function that returns modified url from input url
    :param bands: Only edit specified bands, default is to edit all
    :return: Input dataset
    """
    resolved_paths = measurement_paths(ds)
    if bands is None:
        bands = list(resolved_paths)
    else:
        # remap aliases if present to their canonical name
        bands = list(map(ds.type.canonical_measurement, bands))

    mm = ds.metadata_doc["measurements"]
    for band in bands:
        mm[band]["path"] = edit(resolved_paths[band])
    return ds
