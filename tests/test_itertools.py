"""
Tests for `tqdm.contrib.itertools`.
"""

import itertools as it

from tqdm.std import tqdm
from tqdm.aliases import tproduct, tenumerate, tmap, tzip
import pytest
from io import StringIO
from contextlib import closing


class NoLenIter(object):
    def __init__(self, iterable):
        self._it = iterable

    def __iter__(self):
        for i in self._it:
            yield i


def test_product():
    """Test contrib.itertools.product"""
    with closing(StringIO()) as our_file:
        a = range(9)
        assert list(tproduct(a, a[::-1], file=our_file)) == list(it.product(a, a[::-1]))

        assert list(tproduct(a, NoLenIter(a), file=our_file)) == list(
            it.product(a, NoLenIter(a))
        )


def test_product_with_repeat():
    """Test the case where a repeat argument has been set"""
    with closing(StringIO()) as our_file:
        a = range(9)
        assert list(tproduct(a, repeat=2, file=our_file)) == list(
            it.product(a, repeat=2)
        )


@pytest.mark.parametrize("tqdm_kwargs", [{}, {"tqdm_class": tqdm}])
def test_enumerate(tqdm_kwargs):
    """Test contrib.tenumerate"""
    a = range(9)

    with closing(StringIO()) as our_file:
        assert list(tenumerate(a, file=our_file, **tqdm_kwargs)) == list(enumerate(a))
        assert list(tenumerate(a, 42, file=our_file, **tqdm_kwargs)) == list(
            enumerate(a, 42)
        )
    with closing(StringIO()) as our_file:
        _ = tenumerate(iter(a), file=our_file, **tqdm_kwargs)
        assert "100%" not in our_file.getvalue()
    with closing(StringIO()) as our_file:
        _ = list(tenumerate(iter(a), file=our_file, **tqdm_kwargs))
        assert "100%" in our_file.getvalue()


@pytest.mark.parametrize("tqdm_kwargs", [{}, {"tqdm_class": tqdm}])
def test_zip(tqdm_kwargs):
    """Test contrib.tzip"""
    with closing(StringIO()) as our_file:
        a = range(9)
        b = [i + 1 for i in a]
        gen = tzip(a, b, file=our_file, **tqdm_kwargs)
        assert gen != list(zip(a, b))
        assert list(gen) == list(zip(a, b))


@pytest.mark.parametrize("tqdm_kwargs", [{}, {"tqdm_class": tqdm}])
def test_map(tqdm_kwargs):
    """Test contrib.tmap"""
    with closing(StringIO()) as our_file:
        a = range(9)
        b = [i + 1 for i in a]
        gen = tmap(lambda x: x + 1, a, file=our_file, **tqdm_kwargs)
        assert gen != b
        assert list(gen) == b
