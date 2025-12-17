from typing import Any, Callable, Iterable, Iterator, TypeVar
from .std import tqdm
import itertools
from operator import length_hint

T = TypeVar("T")
R = TypeVar("R")


def tenumerate(
    iterable: Iterable[T],
    start: int = 0,
    total: int | float | None = None,
    tqdm_class: type[tqdm] = tqdm,
    **tqdm_kwargs: Any,
) -> Iterator[tuple[int, T]]:
    """
    Equivalent of builtin `enumerate`.

    Parameters
    ----------
    tqdm_class  : [default: tqdm.std.tqdm].
    """
    return enumerate(tqdm_class(iterable, total=total, **tqdm_kwargs), start)


def tzip(
    iter1: Iterable[T], *iter2plus: Iterable[Any], **tqdm_kwargs: Any
) -> Iterator[tuple[T, ...]]:
    """
    Equivalent of builtin `zip`.

    Parameters
    ----------
    tqdm_class  : [default: tqdm.std.tqdm].
    """
    kwargs = tqdm_kwargs.copy()
    tqdm_class = kwargs.pop("tqdm_class", tqdm)
    for i in zip(tqdm_class(iter1, **kwargs), *iter2plus):
        yield i


def tmap(
    function: Callable[..., R], *sequences: Iterable[Any], **tqdm_kwargs: Any
) -> Iterator[R]:
    """
    Equivalent of builtin `map`.

    Parameters
    ----------
    tqdm_class  : [default: tqdm.std.tqdm].
    """
    for i in tzip(*sequences, **tqdm_kwargs):
        yield function(*i)


def tproduct(*iterables: Iterable[T], **tqdm_kwargs: Any) -> Iterator[tuple[T, ...]]:
    """
    Equivalent of `itertools.product`.

    Parameters
    ----------
    tqdm_class  : [default: tqdm.std.tqdm].
    """
    kwargs = tqdm_kwargs.copy()
    repeat = kwargs.pop("repeat", 1)
    tqdm_class = kwargs.pop("tqdm_class", tqdm)
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
    with tqdm_class(**kwargs) as t:
        it = itertools.product(*iterables, repeat=repeat)
        for val in it:
            yield val
            t.update()


def trange(*args: int, **kwargs: Any) -> tqdm:
    """Shortcut for tqdm(range(*args), **kwargs)."""
    return tqdm(range(*args), **kwargs)
