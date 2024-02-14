"""
Various Dask helpers.
"""

from typing import (
    Any,
    Callable,
    Hashable,
    Iterator,
    MutableMapping,
    Optional,
    Tuple,
    TypeVar,
)

from dask.base import tokenize

T = TypeVar("T")


def tokenize_stream(
    xx: Iterator[T],
    key: Optional[Callable[[str], Hashable]] = None,
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
