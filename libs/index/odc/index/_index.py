import datetime
from typing import Iterator, Tuple
from types import SimpleNamespace

from datacube.index.hl import Doc2Dataset
from datacube.api.query import Query
from datacube.model import Range
from datacube import Datacube
from odc.io.text import parse_yaml


def from_metadata_stream(metadata_stream, index, **kwargs):
    """
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
            yield (None, 'Error: empty doc %s' % (uri))
        else:
            ds, err = doc2ds(metadata, uri)
            if ds is not None:
                yield (ds, None)
            else:
                yield (None, 'Error: %s, %s' % (uri, err))


def parse_doc_stream(doc_stream, on_error=None):
    """Replace doc bytes/strings with parsed dicts.

    doc_stream -- sequence of (uri, doc: byges|string)
    on_error -- uri, doc, exception -> None

    On output doc is replaced with python dict parsed from yaml, or with None
    if parsing error occured.
    """
    for uri, doc in doc_stream:
        try:
            metadata = parse_yaml(doc)
        except Exception as e:
            if on_error is not None:
                on_error(uri, doc, e)
            metadata = None

        yield uri, metadata


def from_yaml_doc_stream(doc_stream, index, logger=None,
                         transform=None,
                         **kwargs):
    """ returns a sequence of tuples where each tuple is either

        (Dataset, None) or (None, error_message)
    """
    def on_parse_error(uri, doc, err):
        if logger is not None:
            logger.error("Failed to parse: %s", uri)
        else:
            print(f"Failed to handle {uri}")

    metadata_stream = parse_doc_stream(doc_stream, on_error=on_parse_error)

    if transform is not None:
        try:
            metadata_stream = map(lambda d: (d[0], transform(d[1])),
                                metadata_stream)
        except Exception as e:
            on_parse_error()

    return from_metadata_stream(metadata_stream, index, **kwargs)


def dataset_count(index, **query):
    return index.datasets.count(**Query(**query).search_terms)


def count_by_year(index, product, min_year=None, max_year=None):
    """ Returns dictionary Int->Int: `year` -> `dataset count for this year`.
        Only non-empty years are reported.
    """

    # TODO: get min/max from datacube properly
    if min_year is None:
        min_year = 1970
    if max_year is None:
        max_year = 2022

    ll = ((year, dataset_count(index, product=product, time=str(year)))
          for year in range(min_year, max_year))

    return {year: c for year, c in ll if c > 0}


def count_by_month(index, product, year):
    """ Return 12 integer tuple
         counts for January, February ... December
    """
    return tuple(dataset_count(index, product=product, time='{}-{:02d}'.format(year, month))
                 for month in range(1, 12+1))


def time_range(begin, end, freq='m'):
    """ Return tuples of datetime objects aligned to boundaries of requested period
    (month is default).

    """
    from pandas import Period
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


def month_range(year: int, month: int, n: int) -> Tuple[datetime.datetime, datetime.datetime]:
    """ Return time range covering n months starting from year, month
        month 1..12
        month can also be negative
        2020, -1 === 2019, 12
    """
    if month < 0:
        return month_range(year-1, 12+month+1, n)

    y2 = year
    m2 = month + n
    if m2 > 12:
        m2 -= 12
        y2 += 1
    dt_eps = datetime.timedelta(microseconds=1)

    return (datetime.datetime(year=year, month=month, day=1),
            datetime.datetime(year=y2, month=m2, day=1) - dt_eps)


def season_range(year: int, season: str) -> Tuple[datetime.datetime, datetime.datetime]:
    """ Season is one of djf, mam, jja, son.

        DJF for year X starts in Dec X-1 and ends in Feb X.
    """
    seasons = dict(
        djf=-1,
        mam=2,
        jja=6,
        son=9)

    start_month = seasons.get(season.lower())
    if start_month is None:
        raise ValueError(f"No such season {season}, valid seasons are: djf,mam,jja,son")
    return month_range(year, start_month, 3)


def chop_query_by_time(q: Query, freq: str = 'm') -> Iterator[Query]:
    """Given a query over longer period of time, chop it up along the time dimension
    into smaller queries each covering a shorter time period (year, month, week or day).
    """
    qq = dict(**q.search_terms)
    time = qq.pop('time')
    if time is None:
        raise ValueError('Need time range in the query')

    for (t0, t1) in time_range(time.begin, time.end, freq=freq):
        yield Query(**qq, time=Range(t0, t1))


def ordered_dss(dc: Datacube, freq: str = 'm', **query):
    """Emulate "order by time" streaming interface for datacube queries.

        Basic idea is to perform a lot of smaller queries (shorter time
        periods), sort results then yield them to the calling code.
    """
    qq = Query(**query)

    for q in chop_query_by_time(qq, freq=freq):
        dss = dc.find_datasets(**q.search_terms)
        dss.sort(key=lambda ds: ds.center_time)
        yield from dss


def bin_dataset_stream(gridspec, dss, persist=None):
    """

    :param gridspec: GridSpec
    :param dss: Sequence of datasets (can be lazy)
    :param persist: Dataset -> SomeThing mapping, defaults to keeping dataset id only

    Return dictionary mapping from (x,y) tile index to object with the following properties

     .idx     - tile index (x,y)
     .geobox  - tile geobox
     .dss     - list of UUIDs, or results of `persist(dataset)` if custom `persist` is supplied
    """
    from warnings import warn

    cells = {}
    geobox_cache = {}

    def default_persist(ds):
        return ds.id

    def register(tile, geobox, val):
        cell = cells.get(tile)
        if cell is None:
            cells[tile] = SimpleNamespace(geobox=geobox, idx=tile, dss=[val])
        else:
            cell.dss.append(val)

    if persist is None:
        persist = default_persist

    for ds in dss:
        ds_val = persist(ds)

        if ds.extent is None:
            warn('Dataset without extent info: %s' % str(ds.id))
            continue

        for tile, geobox in gridspec.tiles_from_geopolygon(ds.extent, geobox_cache=geobox_cache):
            register(tile, geobox, ds_val)

    return cells
