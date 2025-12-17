"""Shared pytest config."""

import sys
from functools import wraps

from pytest import fixture, skip

from tldm import tldm


@fixture(autouse=True)
def pretest_posttest():
    """Fixture for all tests ensuring environment cleanup"""
    try:
        sys.setswitchinterval(1)
    except AttributeError:
        sys.setcheckinterval(100)  # deprecated

    if getattr(tldm, "_instances", False):
        n = len(tldm._instances)
        if n:
            tldm._instances.clear()
            raise OSError(f"{n} `tldm` instances still in existence PRE-test")
    yield
    if getattr(tldm, "_instances", False):
        n = len(tldm._instances)
        if n:
            tldm._instances.clear()
            raise OSError(f"{n} `tldm` instances still in existence POST-test")


def patch_lock(thread=True):
    """decorator replacing tldm's lock with vanilla threading/multiprocessing"""
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
            default_lock = tldm.get_lock()
            try:
                tldm.set_lock(lock)
                return func(*args, **kwargs)
            finally:
                tldm.set_lock(default_lock)

        return inner

    return outer
