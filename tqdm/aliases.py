from .std import tqdm
import itertools
from operator import length_hint


def tenumerate(iterable, start=0, total=None, tqdm_class=tqdm, **tqdm_kwargs):
    """
    Equivalent of builtin `enumerate`.

    Parameters
    ----------
    tqdm_class  : [default: tqdm.std.tqdm].
    """
    return enumerate(tqdm_class(iterable, total=total, **tqdm_kwargs), start)


def tzip(iter1, *iter2plus, **tqdm_kwargs):
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


def tmap(function, *sequences, **tqdm_kwargs):
    """
    Equivalent of builtin `map`.

    Parameters
    ----------
    tqdm_class  : [default: tqdm.std.tqdm].
    """
    for i in tzip(*sequences, **tqdm_kwargs):
        yield function(*i)


def tproduct(*iterables, **tqdm_kwargs):
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
        for i in it:
            yield i
            t.update()


def trange(*args, **kwargs):
    """Shortcut for tqdm(range(*args), **kwargs)."""
    return tqdm(range(*args), **kwargs)
