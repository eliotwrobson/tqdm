"""
Asynchronous progressbar decorator for iterators.
Includes a default `range` iterator printing to `stderr`.

Usage:
>>> from tqdm.asyncio import trange, tqdm
>>> async for i in trange(10):
...     ...
"""

import asyncio
from sys import version_info

from .std import tqdm as std_tqdm


class tqdm_asyncio(std_tqdm):
    """
    Asynchronous-friendly version of tqdm.
    """

    def __init__(self, iterable=None, *args, **kwargs):
        super().__init__(iterable, *args, **kwargs)
        self.iterable_awaitable = False
        if iterable is not None:
            if hasattr(iterable, "__anext__"):
                self.iterable_next = iterable.__anext__
                self.iterable_awaitable = True
            elif hasattr(iterable, "__next__"):
                self.iterable_next = iterable.__next__

    def __aiter__(self):
        return self

    # def __del__(self):
    #     self.close()
    #     if len(tqdm_asyncio._instances) == 0:
    #         if hasattr(tqdm_asyncio, "_lock"):
    #             del tqdm_asyncio._lock
    #         if hasattr(tqdm_asyncio, "monitor") and tqdm_asyncio.monitor is not None:
    #             tqdm_asyncio.monitor.exit()

    async def __anext__(self):
        try:
            if self.iterable_awaitable:
                res = await self.iterable_next()
            else:
                if not hasattr(self, "iterable_iterator"):
                    self.iterable_iterator = iter(self.iterable)
                    self.iterable_next = self.iterable_iterator.__next__
                res = self.iterable_next()
            self.update()
            return res
        except StopIteration:
            self.close()
            raise StopAsyncIteration
        except BaseException:
            self.close()
            raise

    def send(self, *args, **kwargs):
        return self.iterable.send(*args, **kwargs)

    @classmethod
    def as_completed(cls, fs, *, loop=None, timeout=None, total=None, **tqdm_kwargs):
        """
        Wrapper for `asyncio.as_completed`.
        """
        if total is None:
            total = len(fs)
        kwargs = {}
        if version_info[:2] < (3, 10):
            kwargs["loop"] = loop
        yield from cls(
            asyncio.as_completed(fs, timeout=timeout, **kwargs),
            total=total,
            **tqdm_kwargs,
        )

    @classmethod
    async def gather(
        cls,
        *fs,
        loop=None,
        timeout=None,
        total=None,
        return_exceptions=False,
        **tqdm_kwargs,
    ):
        """
        Wrapper for `asyncio.gather`.
        """
        if total is None:
            total = len(fs)

        async def wrap_awaitable(i, f):
            try:
                return i, await f
            except Exception as e:
                if return_exceptions:
                    return i, e
                raise

        async def aiter_as_completed():
            kwargs = {}
            if version_info[:2] < (3, 10):
                kwargs["loop"] = loop
            ifs = [wrap_awaitable(i, f) for i, f in enumerate(fs)]
            for r in asyncio.as_completed(ifs, timeout=timeout, **kwargs):
                yield await r

        res = [f async for f in cls(aiter_as_completed(), total=total, **tqdm_kwargs)]
        return [i for _, i in sorted(res)]
