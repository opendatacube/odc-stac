"""
Generic tools with only standard lib dependencies.
"""
from typing import Iterable, Iterator, Sized, TypeVar

T = TypeVar("T")


class SizedIterable(Sized, Iterable[T]):
    """
    Lazy sequence of known length but no random access.

    Used to passthrough computation to progress callback like tqdm.
    """

    def __init__(self, xx: Iterable[T], n: int) -> None:
        self._xx = iter(xx)
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __iter__(self) -> Iterator[T]:
        yield from self._xx
