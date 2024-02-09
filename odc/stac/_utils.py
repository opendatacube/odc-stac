"""
Generic tools with only standard lib dependencies.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, Iterator, Optional, Sized, TypeVar, Union

T = TypeVar("T")
S = TypeVar("S")


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


def pmap(
    func: Callable[[T], S],
    inputs: Iterable[T],
    pool: Union[ThreadPoolExecutor, int, None],
) -> Iterator[S]:
    """
    Wrapper for ThreadPoolExecutor.map
    """
    if pool is None:
        yield from map(func, inputs)
        return

    if isinstance(pool, int):
        pool = ThreadPoolExecutor(pool)

    with pool as _runner:
        for x in _runner.map(func, inputs):
            yield x
