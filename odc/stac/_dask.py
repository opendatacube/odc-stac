"""
Various Dask helpers.
"""
from typing import Any, Callable, Hashable, Iterator, MutableMapping, Optional, Tuple

import odc.geo.crs
import odc.geo.geobox
from dask.base import normalize_token, tokenize

from ._model import T


# TODO: these classes should just implement __dask_token__ instead
@normalize_token.register(odc.geo.crs.CRS)
def normalize_token_crs(crs):
    return ("odc.geo.crs.CRS", str(crs))


@normalize_token.register(odc.geo.geobox.GeoBox)
def normalize_token_geobox(gbox):
    crs = gbox.crs
    return ("odc.geo.geobox.GeoBox", str(crs), *gbox.shape.yx, *gbox.affine[:6])


@normalize_token.register(odc.geo.geobox.GeoboxTiles)
def normalize_token_gbt(gbt: odc.geo.geobox.GeoboxTiles):
    gbox = gbt.base
    crs = gbox.crs
    return (
        "odc.geo.geobox.GeoboxTiles",
        *gbt.shape.yx,
        str(crs),
        *gbox.shape.yx,
        *gbox.affine[:6],
    )


def tokenize_stream(
    xx: Iterator[T],
    key: Callable[[str], Hashable] = None,
    dsk: Optional[MutableMapping[Hashable, Any]] = None,
) -> Iterator[Tuple[Hashable, T]]:
    if key:
        kx = ((key(tokenize(x)), x) for x in xx)
    else:
        kx = ((tokenize(x), x) for x in xx)

    if dsk is None:
        yield from kx
    else:
        for k, x in kx:
            dsk[k] = x
            yield k, x


def unpack_chunksize(chunk: int, N: int) -> Tuple[int, ...]:
    """
    Compute chunk sizes
    Example: 4, 11 -> (4, 4, 3)
    """
    if chunk >= N or chunk < 0:
        return (N,)

    nb = N // chunk
    last_chunk = N - chunk * nb
    if last_chunk == 0:
        return tuple(chunk for _ in range(nb))

    return tuple(chunk for _ in range(nb)) + (last_chunk,)


def unpack_chunks(
    chunks: Tuple[int, ...], shape: Tuple[int, ...]
) -> Tuple[Tuple[int, ...], ...]:
    """
    Expand chunks
    """
    assert len(chunks) == len(shape)
    return tuple(unpack_chunksize(ch, n) for ch, n in zip(chunks, shape))
