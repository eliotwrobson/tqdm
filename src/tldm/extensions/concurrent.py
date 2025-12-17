"""
Thin wrappers around `concurrent.futures`.
"""

from contextlib import contextmanager
from operator import length_hint
from os import cpu_count

from ..std import tldm as std_tldm
from ..utils import TldmWarning


@contextmanager
def ensure_lock(tldm_class, lock_name=""):
    """get (create if necessary) and then restore `tldm_class`'s lock"""
    old_lock = getattr(tldm_class, "_lock", None)  # don't create a new lock
    lock = old_lock or tldm_class.get_lock()  # maybe create a new lock
    lock = getattr(lock, lock_name, lock)  # maybe subtype
    tldm_class.set_lock(lock)
    yield lock
    if old_lock is None:
        del tldm_class._lock
    else:
        tldm_class.set_lock(old_lock)


def _executor_map(PoolExecutor, fn, *iterables, **tldm_kwargs):
    """
    Implementation of `thread_map` and `process_map`.

    Parameters
    ----------
    tldm_class  : [default: tldm.auto.tldm].
    max_workers  : [default: min(32, cpu_count() + 4)].
    timeout  : [default: None].
    chunksize  : [default: 1].
    lock_name  : [default: "":str].
    """
    kwargs = tldm_kwargs.copy()
    if "total" not in kwargs:
        kwargs["total"] = length_hint(iterables[0])
    tldm_class = kwargs.pop("tldm_class", std_tldm)
    max_workers = kwargs.pop("max_workers", min(32, cpu_count() + 4))
    timeout = kwargs.pop("timeout", None)
    chunksize = kwargs.pop("chunksize", 1)
    lock_name = kwargs.pop("lock_name", "")
    with ensure_lock(tldm_class, lock_name=lock_name) as lk:
        # share lock in case workers are already using `tldm`
        with PoolExecutor(
            max_workers=max_workers, initializer=tldm_class.set_lock, initargs=(lk,)
        ) as ex:
            return list(
                tldm_class(
                    ex.map(fn, *iterables, timeout=timeout, chunksize=chunksize),
                    **kwargs,
                )
            )


def thread_map(fn, *iterables, **tldm_kwargs):
    """
    Equivalent of `list(map(fn, *iterables))`
    driven by `concurrent.futures.ThreadPoolExecutor`.

    Parameters
    ----------
    tldm_class  : optional
        `tldm` class to use for bars [default: tldm.auto.tldm].
    max_workers  : int, optional
        Maximum number of workers to spawn; passed to
        `concurrent.futures.ThreadPoolExecutor.__init__`.
        [default: min(32, cpu_count() + 4)].
    timeout  : int or float, optional
        The iterator raises a TimeoutError if __next()__ is called and the
        result isn't available within the timeout specified from the
        original call to thread_map. [default: None].
    """
    from concurrent.futures import ThreadPoolExecutor

    return _executor_map(ThreadPoolExecutor, fn, *iterables, **tldm_kwargs)


def process_map(fn, *iterables, **tldm_kwargs):
    """
    Equivalent of `list(map(fn, *iterables))`
    driven by `concurrent.futures.ProcessPoolExecutor`.

    Parameters
    ----------
    tldm_class  : optional
        `tldm` class to use for bars [default: tldm.auto.tldm].
    max_workers  : int, optional
        Maximum number of workers to spawn; passed to
        `concurrent.futures.ProcessPoolExecutor.__init__`.
        [default: min(32, cpu_count() + 4)].
    timeout  : int or float, optional
        The iterator raises a TimeoutError if __next()__ is called and the
        result isn't available within the timeout specified from the
        original call to process_map. [default: None].
    chunksize  : int, optional
        Size of chunks sent to worker processes; passed to
        `concurrent.futures.ProcessPoolExecutor.map`. [default: 1].
    lock_name  : str, optional
        Member of `tldm_class.get_lock()` to use [default: mp_lock].
    """
    from concurrent.futures import ProcessPoolExecutor

    if iterables and "chunksize" not in tldm_kwargs:
        # default `chunksize=1` has poor performance for large iterables
        # (most time spent dispatching items to workers).
        longest_iterable_len = max(map(length_hint, iterables))
        if longest_iterable_len > 1000:
            from warnings import warn

            warn(
                "Iterable length %d > 1000 but `chunksize` is not set."
                " This may seriously degrade multiprocess performance."
                " Set `chunksize=1` or more." % longest_iterable_len,
                TldmWarning,
                stacklevel=2,
            )
    if "lock_name" not in tldm_kwargs:
        tldm_kwargs = tldm_kwargs.copy()
        tldm_kwargs["lock_name"] = "mp_lock"
    return _executor_map(ProcessPoolExecutor, fn, *iterables, **tldm_kwargs)
