# pylint: disable=missing-function-docstring, missing-module-docstring
from ._utils import SizedIterable


def test_sized_iterator():
    assert len(SizedIterable([], 0)) == 0
    assert len(SizedIterable(iter([1, 2, 3]), 3)) == 3

    xx = SizedIterable(range(5), 5)
    assert len(xx) == 5
    assert list(xx) == list(range(5))

    xx = SizedIterable(range(100, 102), 2)
    a, b = xx
    assert a, b == (100, 101)
