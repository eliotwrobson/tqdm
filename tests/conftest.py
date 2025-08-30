"""Shared pytest config."""

import sys

from pytest import fixture, skip

from tqdm import tqdm

from functools import wraps


@fixture(autouse=True)
def pretest_posttest():
    """Fixture for all tests ensuring environment cleanup"""
    try:
        sys.setswitchinterval(1)
    except AttributeError:
        sys.setcheckinterval(100)  # deprecated

    if getattr(tqdm, "_instances", False):
        n = len(tqdm._instances)
        if n:
            tqdm._instances.clear()
            raise EnvironmentError(f"{n} `tqdm` instances still in existence PRE-test")
    yield
    if getattr(tqdm, "_instances", False):
        n = len(tqdm._instances)
        if n:
            tqdm._instances.clear()
            raise EnvironmentError(f"{n} `tqdm` instances still in existence POST-test")


def patch_lock(thread=True):
    """decorator replacing tqdm's lock with vanilla threading/multiprocessing"""
    try:
        if thread:
            from threading import RLock
        else:
            from multiprocessing import RLock
        lock = RLock()
    except (ImportError, OSError) as err:
        skip(str(err))

    def outer(func):
        """actual decorator"""

        @wraps(func)
        def inner(*args, **kwargs):
            """set & reset lock even if exceptions occur"""
            default_lock = tqdm.get_lock()
            try:
                tqdm.set_lock(lock)
                return func(*args, **kwargs)
            finally:
                tqdm.set_lock(default_lock)

        return inner

    return outer
