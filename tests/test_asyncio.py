"""Tests `tldm.asyncio`."""

import asyncio
from contextlib import closing
from functools import partial
from io import StringIO
from sys import platform
from time import time

from pytest import mark, raises

from tldm.extensions.asyncio import tldm_asyncio

tldm = partial(tldm_asyncio, miniters=0, mininterval=0)
as_completed = partial(tldm_asyncio.as_completed, miniters=0, mininterval=0)
gather = partial(tldm_asyncio.gather, miniters=0, mininterval=0)


def count(start=0, step=1):
    i = start
    while True:
        new_start = yield i
        if new_start is None:
            i += step
        else:
            i = new_start


async def acount(*args, **kwargs):
    for i in count(*args, **kwargs):
        yield i


@mark.asyncio
async def test_break():
    """Test asyncio break"""
    pbar = tldm(count())
    async for _ in pbar:
        break
    pbar.close()


@mark.asyncio
async def test_complete_bar_on_early_finish(capsys):
    """Test completing the bar on early exit."""
    with tldm(
        count(),
        total=10,
        miniters=0,
        mininterval=0,
        complete_bar_on_early_finish=True,
    ) as pbar:
        async for i in pbar:
            if i == 4:
                break
    _, err = capsys.readouterr()
    assert "10/10" in err


@mark.asyncio
async def test_no_complete_bar_on_exception(capsys):
    """Test not completing the bar on exception."""
    with raises(RuntimeError):
        with tldm(
            count(),
            total=10,
            miniters=0,
            mininterval=0,
            complete_bar_on_early_finish=True,
        ) as pbar:
            async for i in pbar:
                if i == 4:
                    raise RuntimeError("boom")
    _, err = capsys.readouterr()
    assert "10/10" not in err


@mark.asyncio
async def test_generators(capsys):
    """Test asyncio generators"""
    with tldm(count(), desc="counter") as pbar:
        async for i in pbar:
            if i >= 8:
                break
    _, err = capsys.readouterr()
    assert "9it" in err

    acounter = acount()
    try:
        with tldm(acounter, desc="async_counter") as pbar:
            async for i in pbar:
                if i >= 8:
                    break
    finally:
        await acounter.aclose()
    _, err = capsys.readouterr()
    assert "9it" in err


@mark.asyncio
async def test_coroutines():
    """Test asyncio coroutine.send"""
    with closing(StringIO()) as our_file:
        with tldm(count(), file=our_file) as pbar:
            async for i in pbar:
                if i == 9:
                    pbar.send(-10)
                elif i < 0:
                    assert i == -9
                    break
        assert "10it" in our_file.getvalue()


@mark.slow
@mark.asyncio
@mark.parametrize("tol", [0.2 if platform.startswith("darwin") else 0.1])
async def test_as_completed(capsys, tol):
    """Test asyncio as_completed"""
    for retry in range(3):
        t = time()
        skew = time() - t
        for i in as_completed([asyncio.sleep(0.01 * i) for i in range(30, 0, -1)]):
            await i
        t = time() - t - 2 * skew
        try:
            assert 0.3 * (1 - tol) < t < 0.3 * (1 + tol), t
            _, err = capsys.readouterr()
            assert "30/30" in err
        except AssertionError:
            if retry == 2:
                raise


async def double(i):
    return i * 2


@mark.asyncio
async def test_gather(capsys):
    """Test asyncio gather"""
    expected = list(range(0, 30 * 2, 2))
    res = await gather(*map(double, range(30)))
    _, err = capsys.readouterr()
    assert "30/30" in err
    assert res == expected

    res = await gather(*map(double, range(30)), double(time), return_exceptions=True)
    _, err = capsys.readouterr()
    assert "31/31" in err
    assert res[:-1] == expected
    assert isinstance(res[-1], TypeError)

    with raises(TypeError):
        await gather(*map(double, range(30)), double(time))
