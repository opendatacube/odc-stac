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
