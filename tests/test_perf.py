import sys
from contextlib import contextmanager

# Use relative/cpu timer to have reliable timings when there is a sudden load
from time import process_time, sleep, time

from pytest import importorskip, mark, skip

from tldm import tldm, trange

from .conftest import patch_lock


def cpu_sleep(t):
    """Sleep the given amount of cpu time"""
    start = process_time()
    while (process_time() - start) < t:
        pass


def checkCpuTime(sleeptime=0.2):
    """Check if cpu time works correctly"""
    if checkCpuTime.passed:
        return True
    # First test that sleeping does not consume cputime
    start1 = process_time()
    sleep(sleeptime)
    t1 = process_time() - start1

    # secondly check by comparing to cpusleep (where we actually do something)
    start2 = process_time()
    cpu_sleep(sleeptime)
    t2 = process_time() - start2

    if abs(t1) < 0.0001 and t1 < t2 / 10:
        checkCpuTime.passed = True
        return True
    skip("cpu time not reliable on this machine")


checkCpuTime.passed = False


@contextmanager
def relative_timer():
    """yields a context timer function which stops ticking on exit"""
    start = process_time()

    def elapser():
        return process_time() - start

    yield lambda: elapser()
    spent = elapser()

    def elapser():  # NOQA
        return spent


@contextmanager
def suppress_output():
    """Suppress stdout and stderr to avoid large console output on test failures"""
    import io

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def simple_progress(
    iterable=None,
    total=None,
    file=sys.stdout,
    desc="",
    leave=False,
    miniters=1,
    mininterval=0.1,
    width=60,
):
    """Simple progress bar reproducing tldm's major features"""
    n = [0]  # use a closure
    start_t = [time()]
    last_n = [0]
    last_t = [0]
    if iterable is not None:
        total = len(iterable)

    def format_interval(t):
        mins, s = divmod(int(t), 60)
        h, m = divmod(mins, 60)
        return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

    def update_and_print(i=1):
        n[0] += i
        if (n[0] - last_n[0]) >= miniters:
            last_n[0] = n[0]

            if (time() - last_t[0]) >= mininterval:
                last_t[0] = time()  # last_t[0] == current time

                spent = last_t[0] - start_t[0]
                spent_fmt = format_interval(spent)
                rate = n[0] / spent if spent > 0 else 0
                rate_fmt = "%.2fs/it" % (1.0 / rate) if 0.0 < rate < 1.0 else "%.2fit/s" % rate

                frac = n[0] / total
                percentage = int(frac * 100)
                eta = (total - n[0]) / rate if rate > 0 else 0
                eta_fmt = format_interval(eta)

                # full_bar = "#" * int(frac * width)
                barfill = " " * int((1.0 - frac) * width)
                bar_length, frac_bar_length = divmod(int(frac * width * 10), 10)
                full_bar = "#" * bar_length
                frac_bar = chr(48 + frac_bar_length) if frac_bar_length else " "

                file.write(
                    "\r%s %i%%|%s%s%s| %i/%i [%s<%s, %s]"
                    % (
                        desc,
                        percentage,
                        full_bar,
                        frac_bar,
                        barfill,
                        n[0],
                        total,
                        spent_fmt,
                        eta_fmt,
                        rate_fmt,
                    )
                )

                if n[0] == total and leave:
                    file.write("\n")
                file.flush()

    def update_and_yield():
        for elt in iterable:
            yield elt
            update_and_print()

    update_and_print(0)
    if iterable is not None:
        return update_and_yield()
    return update_and_print


def assert_performance(thresh, name_left, time_left, name_right, time_right):
    """raises if time_left > thresh * time_right"""
    if time_left > thresh * time_right:
        raise ValueError(
            f"{name_left}: {time_left:f}, {name_right}: {time_right:f}"
            f", ratio {time_left / time_right:f} > {thresh:f}"
        )


@mark.flaky(reruns=3)
@mark.xfail(reason="Performance tests are flaky on CI runners", strict=False)
def test_iter_basic_overhead():
    """Test overhead of iteration based tldm"""
    checkCpuTime()
    total = int(1e6)

    with suppress_output():
        a = 0
        with trange(total) as t, relative_timer() as time_tldm:
            for i in t:
                a += i
        assert a == (total**2 - total) / 2.0

        a = 0
        with relative_timer() as time_bench:
            for i in range(total):
                a += i
                sys.stdout.write(str(a))

    assert_performance(3, "trange", time_tldm(), "range", time_bench())


@mark.flaky(reruns=3)
@mark.xfail(reason="Performance tests are flaky on CI runners", strict=False)
def test_manual_basic_overhead():
    """Test overhead of manual tldm"""
    checkCpuTime()
    total = int(1e6)

    with suppress_output():
        with tldm(total=total * 10, leave=True) as t:
            a = 0
            with relative_timer() as time_tldm:
                for i in range(total):
                    a += i
                    t.update(10)

        a = 0
        with relative_timer() as time_bench:
            for i in range(total):
                a += i
                sys.stdout.write(str(a))

    assert_performance(5, "tldm", time_tldm(), "range", time_bench())


def worker(total, blocking=True):
    def incr_bar(x):
        for _ in trange(
            total,
            lock_args=None if blocking else (False,),
            miniters=1,
            mininterval=0,
            maxinterval=0,
        ):
            pass
        return x + 1

    return incr_bar


@patch_lock(thread=True)
@mark.flaky(reruns=3)
@mark.skip(reason="flaky test, needs investigation")
def test_lock_args():
    """Test overhead of nonblocking threads"""
    checkCpuTime()
    ThreadPoolExecutor = importorskip("concurrent.futures").ThreadPoolExecutor

    total = 16
    subtotal = 10000

    with ThreadPoolExecutor() as pool:
        sys.stderr.write("block ... ")
        sys.stderr.flush()
        with relative_timer() as time_tldm:
            res = list(pool.map(worker(subtotal, True), range(total)))
            assert sum(res) == sum(range(total)) + total
        sys.stderr.write("noblock ... ")
        sys.stderr.flush()
        with relative_timer() as time_noblock:
            res = list(pool.map(worker(subtotal, False), range(total)))
            assert sum(res) == sum(range(total)) + total

    assert_performance(0.5, "noblock", time_noblock(), "tldm", time_tldm())


@mark.flaky(reruns=10)
@mark.xfail(reason="Performance tests are flaky on CI runners", strict=False)
def test_iter_overhead_hard():
    """Test overhead of iteration based tldm (hard)"""
    checkCpuTime()
    total = int(1e5)

    with suppress_output():
        a = 0
        with trange(total, leave=True, miniters=1, mininterval=0, maxinterval=0) as t:
            with relative_timer() as time_tldm:
                for i in t:
                    a += i
        assert a == (total**2 - total) / 2.0

        a = 0
        with relative_timer() as time_bench:
            for i in range(total):
                a += i
                sys.stdout.write(("%i" % a) * 40)

    assert_performance(130, "trange", time_tldm(), "range", time_bench())


@mark.flaky(reruns=10)
@mark.xfail(reason="Performance tests are flaky on CI runners", strict=False)
def test_manual_overhead_hard():
    """Test overhead of manual tldm (hard)"""
    checkCpuTime()
    total = int(1e5)

    with suppress_output():
        with tldm(total=total * 10, leave=True, miniters=1, mininterval=0, maxinterval=0) as t:
            a = 0
            with relative_timer() as time_tldm:
                for i in range(total):
                    a += i
                    t.update(10)

        a = 0
        with relative_timer() as time_bench:
            for i in range(total):
                a += i
                sys.stdout.write(("%i" % a) * 40)

    assert_performance(130, "tldm", time_tldm(), "range", time_bench())


@mark.flaky(reruns=10)
@mark.xfail(reason="Performance tests are flaky on CI runners", strict=False)
def test_iter_overhead_simplebar_hard():
    """Test overhead of iteration based tldm vs simple progress bar (hard)"""
    checkCpuTime()
    total = int(1e4)

    with suppress_output():
        a = 0
        with trange(total, leave=True, miniters=1, mininterval=0, maxinterval=0) as t:
            with relative_timer() as time_tldm:
                for i in t:
                    a += i
        assert a == (total**2 - total) / 2.0

        a = 0
        s = simple_progress(range(total), leave=True, miniters=1, mininterval=0)
        with relative_timer() as time_bench:
            for i in s:
                a += i

    assert_performance(10, "trange", time_tldm(), "simple_progress", time_bench())


@mark.flaky(reruns=10)
@mark.xfail(reason="Performance tests are flaky on CI runners", strict=False)
def test_manual_overhead_simplebar_hard():
    """Test overhead of manual tldm vs simple progress bar (hard)"""
    checkCpuTime()
    total = int(1e4)

    with suppress_output():
        with tldm(total=total * 10, leave=True, miniters=1, mininterval=0, maxinterval=0) as t:
            a = 0
            with relative_timer() as time_tldm:
                for i in range(total):
                    a += i
                    t.update(10)

        simplebar_update = simple_progress(total=total * 10, leave=True, miniters=1, mininterval=0)
        a = 0
        with relative_timer() as time_bench:
            for i in range(total):
                a += i
                simplebar_update(10)

    assert_performance(12, "tldm", time_tldm(), "simple_progress", time_bench())
