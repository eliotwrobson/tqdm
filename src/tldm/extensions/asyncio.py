"""
Asynchronous progressbar decorator for iterators.
"""

import asyncio

from ..std import tldm as std_tldm


class tldm_asyncio(std_tldm):
    """
    Asynchronous-friendly version of tldm.
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
    #     if len(tldm_asyncio._instances) == 0:
    #         if hasattr(tldm_asyncio, "_lock"):
    #             del tldm_asyncio._lock
    #         if hasattr(tldm_asyncio, "monitor") and tldm_asyncio.monitor is not None:
    #             tldm_asyncio.monitor.exit()

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
            self._close_with_exception = False
            self.close()
            raise StopAsyncIteration from None
        except BaseException:
            self._close_with_exception = True
            self.close()
            raise

    def send(self, *args, **kwargs):
        return self.iterable.send(*args, **kwargs)

    @classmethod
    def as_completed(cls, fs, *, loop=None, timeout=None, total=None, **tldm_kwargs):
        """
        Wrapper for `asyncio.as_completed`.
        """
        if total is None:
            total = len(fs)
        yield from cls(
            asyncio.as_completed(fs, timeout=timeout),
            total=total,
            **tldm_kwargs,
        )

    @classmethod
    async def gather(
        cls,
        *fs,
        loop=None,
        timeout=None,
        total=None,
        return_exceptions=False,
        **tldm_kwargs,
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
            ifs = [wrap_awaitable(i, f) for i, f in enumerate(fs)]
            for r in asyncio.as_completed(ifs, timeout=timeout):
                yield await r

        res = [f async for f in cls(aiter_as_completed(), total=total, **tldm_kwargs)]
        return [i for _, i in sorted(res)]
