import itertools
from collections.abc import Callable, Iterable, Iterator
from operator import length_hint
from typing import Any, TypeVar

from .std import tldm

T = TypeVar("T")
R = TypeVar("R")


def tenumerate(
    iterable: Iterable[T],
    start: int = 0,
    total: int | float | None = None,
    tldm_class: type[tldm] = tldm,
    **tldm_kwargs: Any,
) -> Iterator[tuple[int, T]]:
    """
    Equivalent of builtin `enumerate`.

    Parameters
    ----------
    tldm_class  : [default: tldm.std.tldm].
    """
    return enumerate(tldm_class(iterable, total=total, **tldm_kwargs), start)


def tzip(
    iter1: Iterable[T], *iter2plus: Iterable[Any], **tldm_kwargs: Any
) -> Iterator[tuple[T, ...]]:
    """
    Equivalent of builtin `zip`.

    Parameters
    ----------
    tldm_class  : [default: tldm.std.tldm].
    """
    kwargs = tldm_kwargs.copy()
    tldm_class = kwargs.pop("tldm_class", tldm)
    yield from zip(tldm_class(iter1, **kwargs), *iter2plus)


def tmap(function: Callable[..., R], *sequences: Iterable[Any], **tldm_kwargs: Any) -> Iterator[R]:
    """
    Equivalent of builtin `map`.

    Parameters
    ----------
    tldm_class  : [default: tldm.std.tldm].
    """
    for i in tzip(*sequences, **tldm_kwargs):
        yield function(*i)


def tproduct(*iterables: Iterable[T], **tldm_kwargs: Any) -> Iterator[tuple[T, ...]]:
    """
    Equivalent of `itertools.product`.

    Parameters
    ----------
    tldm_class  : [default: tldm.std.tldm].
    """
    kwargs = tldm_kwargs.copy()
    repeat = kwargs.pop("repeat", 1)
    tldm_class = kwargs.pop("tldm_class", tldm)
    try:
        lens = list(map(length_hint, iterables))
    except TypeError:
        total = None
    else:
        total = 1
        for i in lens:
            total *= i
        total = total**repeat
        kwargs.setdefault("total", total)
    with tldm_class(**kwargs) as t:
        it = itertools.product(*iterables, repeat=repeat)
        for val in it:
            yield val
            t.update()


def trange(*args: int, **kwargs: Any) -> tldm:
    """Shortcut for tldm(range(*args), **kwargs)."""
    return tldm(range(*args), **kwargs)
