import contextlib
import csv
import os
import re
import sys
from contextlib import closing, contextmanager
from io import (
    BytesIO,
    IOBase,  # to support unicode strings
    StringIO,
)
from warnings import catch_warnings, simplefilter

from pytest import importorskip, mark

from tldm import tldm, trange
from tldm.utils import ObjectWrapper, TldmWarning, format_interval

from .conftest import patch_lock


class DeprecationError(Exception):
    pass


nt_and_no_colorama = False
if os.name == "nt":
    try:
        import colorama  # NOQA
    except ImportError:
        nt_and_no_colorama = True

# Regex definitions
# List of control characters
CTRLCHR = [r"\r", r"\n", r"\x1b\[A"]  # Need to escape [ for regex
# Regular expressions compilation
RE_rate = re.compile(r"[^\d](\d[.\d]+)it/s")
RE_ctrlchr = re.compile("(%s)" % "|".join(CTRLCHR))  # Match control chars
RE_ctrlchr_excl = re.compile("|".join(CTRLCHR))  # Match and exclude ctrl chars
RE_pos = re.compile(r"([\r\n]+((pos\d+) bar:\s+\d+%|\s{3,6})?[^\r\n]*)")


class DummyTldmFile(ObjectWrapper):
    """Dummy file-like that will write to tldm"""

    def __init__(self, wrapped):
        super().__init__(wrapped)
        self._buf = []

    def write(self, x, nolock=False):
        nl = b"\n" if isinstance(x, bytes) else "\n"
        pre, sep, post = x.rpartition(nl)
        if sep:
            blank = type(nl)()
            tldm.write(
                blank.join(self._buf + [pre, sep]),
                end=blank,
                file=self._wrapped,
                nolock=nolock,
            )
            self._buf = [post]
        else:
            self._buf.append(x)

    def __del__(self):
        if self._buf:
            blank = type(self._buf[0])()
            with contextlib.suppress(OSError, ValueError):
                tldm.write(blank.join(self._buf), end=blank, file=self._wrapped)


def pos_line_diff(res_list, expected_list, raise_nonempty=True):
    """
    Return differences between two bar output lists.
    To be used with `RE_pos`
    """
    res = [
        (r, e)
        for r, e in zip(res_list, expected_list)
        for pos in [len(e) - len(e.lstrip("\n"))]  # bar position
        if r != e  # simple comparison
        if not r.startswith(e)  # start matches
        or not (
            # move up at end (maybe less due to closing bars)
            any(
                r.endswith(end + i * "\x1b[A")
                for i in range(pos + 1)
                for end in [
                    "]",  # bar
                    "  ",  # cleared
                    "%",  # percentage only format
                ]
            )
            or "100%" in r  # completed bar
            or r == "\n"
        )  # final bar
        or r[(-1 - pos) * len("\x1b[A") :] == "\x1b[A"
    ]  # too many moves up
    if raise_nonempty and (res or len(res_list) != len(expected_list)):
        if len(res_list) < len(expected_list):
            res.extend([(None, e) for e in expected_list[len(res_list) :]])
        elif len(res_list) > len(expected_list):
            res.extend([(r, None) for r in res_list[len(expected_list) :]])
        raise AssertionError("Got => Expected\n" + "\n".join("%r => %r" % i for i in res))
    return res


class DiscreteTimer:
    """Virtual discrete time manager, to precisely control time for tests"""

    def __init__(self):
        self.t = 0.0

    def sleep(self, t):
        """Sleep = increment the time counter (almost no CPU used)"""
        self.t += t

    def time(self):
        """Get the current time"""
        return self.t


def cpu_timify(t, timer=None):
    """Force tldm to use the specified timer instead of system-wide time()"""
    if timer is None:
        timer = DiscreteTimer()
    t._time = timer.time
    t._sleep = timer.sleep
    t.start_t = t.last_print_t = t._time()
    return timer


class UnicodeIO(IOBase):
    """Unicode version of StringIO"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoding = "U8"  # io.StringIO supports unicode, but no encoding
        self.text = ""
        self.cursor = 0

    def __len__(self):
        return len(self.text)

    def seek(self, offset):
        self.cursor = offset

    def tell(self):
        return self.cursor

    def write(self, s):
        self.text = self.text[: self.cursor] + s + self.text[self.cursor + len(s) :]
        self.cursor += len(s)

    def read(self, n=-1):
        _cur = self.cursor
        self.cursor = len(self) if n < 0 else min(_cur + n, len(self))
        return self.text[_cur : self.cursor]

    def getvalue(self):
        return self.text


def get_bar(all_bars, i=None):
    """Get a specific update from a whole bar traceback"""
    # Split according to any used control characters
    bars_split = RE_ctrlchr_excl.split(all_bars)
    bars_split = list(filter(None, bars_split))  # filter out empty splits
    return bars_split if i is None else bars_split[i]


def progressbar_rate(bar_str):
    return float(RE_rate.search(bar_str).group(1))


def squash_ctrlchars(s):
    """Apply control characters in a string just like a terminal display"""
    curline = 0
    lines = [""]  # state of fake terminal
    for nextctrl in filter(None, RE_ctrlchr.split(s)):
        # apply control chars
        if nextctrl == "\r":
            # go to line beginning (simplified here: just empty the string)
            lines[curline] = ""
        elif nextctrl == "\n":
            if curline >= len(lines) - 1:
                # wrap-around creates newline
                lines.append("")
            # move cursor down
            curline += 1
        elif nextctrl == "\x1b[A":
            # move cursor up
            if curline > 0:
                curline -= 1
            else:
                raise ValueError("Cannot go further up")
        else:
            # print message on current line
            lines[curline] += nextctrl
    return lines


def test_all_defaults() -> None:
    """Test default kwargs"""
    with closing(UnicodeIO()) as our_file, tldm(range(10), file=our_file) as progressbar:
        assert len(progressbar) == 10
        for _ in progressbar:
            pass
    # restore stdout/stderr output for `nosetest` interface
    # try:
    #     sys.stderr.write('\x1b[A')
    # except:
    #     pass
    sys.stderr.write("\rTest default kwargs ... ")


class WriteTypeChecker(BytesIO):
    """File-like to assert the expected type is written"""

    def __init__(self, expected_type):
        super().__init__()
        self.expected_type = expected_type

    def write(self, s):
        assert isinstance(s, self.expected_type)


def test_native_string_io_for_default_file() -> None:
    """Native strings written to unspecified files"""
    stderr = sys.stderr
    try:
        sys.stderr = WriteTypeChecker(expected_type=str)
        for _ in tldm(range(3)):
            pass
        sys.stderr.encoding = None  # py2 behaviour
        for _ in tldm(range(3)):
            pass
    finally:
        sys.stderr = stderr


def test_unicode_string_io_for_specified_file() -> None:
    """Unicode strings written to specified files"""
    for _ in tldm(range(3), file=WriteTypeChecker(expected_type=str)):
        pass


def test_iterate_over_csv_rows() -> None:
    """Test csv iterator"""
    # Create a test csv pseudo file
    with closing(StringIO()) as test_csv_file:
        writer = csv.writer(test_csv_file)
        for _ in range(3):
            writer.writerow(["test"] * 3)
        test_csv_file.seek(0)

        # Test that nothing fails if we iterate over rows
        reader = csv.DictReader(test_csv_file, fieldnames=("row1", "row2", "row3"))
        with closing(StringIO()) as our_file:
            for _ in tldm(reader, file=our_file):
                pass


def test_file_output() -> None:
    """Test output to arbitrary file-like objects"""
    with closing(StringIO()) as our_file:
        for i in tldm(range(3), file=our_file):
            if i == 1:
                our_file.seek(0)
                assert "0/3" in our_file.read()


def test_leave_option() -> None:
    """Test `leave=True` always prints info about the last iteration"""
    with closing(StringIO()) as our_file:
        for _ in tldm(range(3), file=our_file, leave=True):
            pass
        res = our_file.getvalue()
        assert "| 3/3 " in res
        assert res[-1] == "\n"  # not '\r'

    with closing(StringIO()) as our_file2:
        for _ in tldm(range(3), file=our_file2, leave=False):
            pass
        assert "| 3/3 " not in our_file2.getvalue()


def test_trange() -> None:
    """Test trange"""
    with closing(StringIO()) as our_file:
        for _ in trange(3, file=our_file, leave=True):
            pass
        assert "| 3/3 " in our_file.getvalue()

    with closing(StringIO()) as our_file2:
        for _ in trange(3, file=our_file2, leave=False):
            pass
        assert "| 3/3 " not in our_file2.getvalue()


def test_min_interval() -> None:
    """Test mininterval"""
    with closing(StringIO()) as our_file:
        for _ in tldm(range(3), file=our_file, mininterval=1e-10):
            pass
        assert "  0%|          | 0/3 [00:00<" in our_file.getvalue()


def test_max_interval() -> None:
    """Test maxinterval"""
    total = 100
    bigstep = 10
    smallstep = 5

    # Test without maxinterval
    timer = DiscreteTimer()
    with closing(StringIO()) as our_file:
        with closing(StringIO()) as our_file2:
            # with maxinterval but higher than loop sleep time
            t = tldm(
                total=total,
                file=our_file,
                miniters=None,
                mininterval=0,
                smoothing=1,
                maxinterval=1e-2,
            )
            cpu_timify(t, timer)

            # without maxinterval
            t2 = tldm(
                total=total,
                file=our_file2,
                miniters=None,
                mininterval=0,
                smoothing=1,
                maxinterval=None,
            )
            cpu_timify(t2, timer)

            assert t.dynamic_miniters
            assert t2.dynamic_miniters

            # Increase 10 iterations at once
            t.update(bigstep)
            t2.update(bigstep)
            # The next iterations should not trigger maxinterval (step 10)
            for _ in range(4):
                t.update(smallstep)
                t2.update(smallstep)
                timer.sleep(1e-5)
            t.close()  # because PyPy doesn't gc immediately
            t2.close()  # as above

            assert "25%" not in our_file2.getvalue()
        assert "25%" not in our_file.getvalue()

    # Test with maxinterval effect
    timer = DiscreteTimer()
    with (
        closing(StringIO()) as our_file,
        tldm(
            total=total,
            file=our_file,
            miniters=None,
            mininterval=0,
            smoothing=1,
            maxinterval=1e-4,
        ) as t,
    ):
        cpu_timify(t, timer)

        # Increase 10 iterations at once
        t.update(bigstep)
        # The next iterations should trigger maxinterval (step 5)
        for _ in range(4):
            t.update(smallstep)
            timer.sleep(1e-2)

        assert "25%" in our_file.getvalue()

    # Test iteration based tldm with maxinterval effect
    timer = DiscreteTimer()
    with closing(StringIO()) as our_file:
        with tldm(
            range(total),
            file=our_file,
            miniters=None,
            mininterval=1e-5,
            smoothing=1,
            maxinterval=1e-4,
        ) as t2:
            cpu_timify(t2, timer)

            for i in t2:
                if i >= (bigstep - 1) and ((i - (bigstep - 1)) % smallstep) == 0:
                    timer.sleep(1e-2)
                if i >= 3 * bigstep:
                    break

        assert "15%" in our_file.getvalue()

    # Test different behavior with and without mininterval
    timer = DiscreteTimer()
    total = 1000
    mininterval = 0.1
    maxinterval = 10
    with (
        closing(StringIO()) as our_file,
        tldm(
            total=total,
            file=our_file,
            miniters=None,
            smoothing=1,
            mininterval=mininterval,
            maxinterval=maxinterval,
        ) as tm1,
        tldm(
            total=total,
            file=our_file,
            miniters=None,
            smoothing=1,
            mininterval=0,
            maxinterval=maxinterval,
        ) as tm2,
    ):
        cpu_timify(tm1, timer)
        cpu_timify(tm2, timer)

        # Fast iterations, check if dynamic_miniters triggers
        timer.sleep(mininterval)  # to force update for t1
        tm1.update(total / 2)
        tm2.update(total / 2)
        assert int(tm1.miniters) == tm2.miniters == total / 2

        # Slow iterations, check different miniters if mininterval
        timer.sleep(maxinterval * 2)
        tm1.update(total / 2)
        tm2.update(total / 2)
        res = [tm1.miniters, tm2.miniters]
        assert res == [
            (total / 2) * mininterval / (maxinterval * 2),
            (total / 2) * maxinterval / (maxinterval * 2),
        ]

    # Same with iterable based tldm
    timer1 = DiscreteTimer()  # need 2 timers for each bar because zip not work
    timer2 = DiscreteTimer()
    total = 100
    mininterval = 0.1
    maxinterval = 10
    with closing(StringIO()) as our_file:
        t1 = tldm(
            range(total),
            file=our_file,
            miniters=None,
            smoothing=1,
            mininterval=mininterval,
            maxinterval=maxinterval,
        )
        t2 = tldm(
            range(total),
            file=our_file,
            miniters=None,
            smoothing=1,
            mininterval=0,
            maxinterval=maxinterval,
        )

        cpu_timify(t1, timer1)
        cpu_timify(t2, timer2)

        for i in t1:
            if i == ((total / 2) - 2):
                timer1.sleep(mininterval)
            if i == (total - 1):
                timer1.sleep(maxinterval * 2)

        for i in t2:
            if i == ((total / 2) - 2):
                timer2.sleep(mininterval)
            if i == (total - 1):
                timer2.sleep(maxinterval * 2)

        assert t1.miniters == 0.255
        assert t2.miniters == 0.5

        t1.close()
        t2.close()


def test_delay() -> None:
    """Test delay"""
    timer = DiscreteTimer()
    with closing(StringIO()) as our_file:
        t = tldm(total=2, file=our_file, leave=True, delay=3)
        cpu_timify(t, timer)
        timer.sleep(2)
        t.update(1)
        assert not our_file.getvalue()
        timer.sleep(2)
        t.update(1)
        assert our_file.getvalue()
        t.close()


def test_min_iters() -> None:
    """Test miniters"""
    with closing(StringIO()) as our_file:
        for _ in tldm(range(3), file=our_file, leave=True, mininterval=0, miniters=2):
            pass

        out = our_file.getvalue()
        assert "| 0/3 " in out
        assert "| 1/3 " not in out
        assert "| 2/3 " in out
        assert "| 3/3 " in out

    with closing(StringIO()) as our_file:
        for _ in tldm(range(3), file=our_file, leave=True, mininterval=0, miniters=1):
            pass

        out = our_file.getvalue()
        assert "| 0/3 " in out
        assert "| 1/3 " in out
        assert "| 2/3 " in out
        assert "| 3/3 " in out


def test_dynamic_min_iters() -> None:
    """Test purely dynamic miniters (and manual updates and __del__)"""
    with closing(StringIO()) as our_file:
        total = 10
        t = tldm(total=total, file=our_file, miniters=None, mininterval=0, smoothing=1)

        t.update()
        # Increase 3 iterations
        t.update(3)
        # The next two iterations should be skipped because of dynamic_miniters
        t.update()
        t.update()
        # The third iteration should be displayed
        t.update()

        out = our_file.getvalue()
        assert t.dynamic_miniters
        del t  # simulate immediate del gc

    assert "  0%|          | 0/10 [00:00<" in out
    assert "40%" in out
    assert "50%" not in out
    assert "60%" not in out
    assert "70%" in out

    # Check with smoothing=0, miniters should be set to max update seen so far
    with closing(StringIO()) as our_file:
        total = 10
        t = tldm(total=total, file=our_file, miniters=None, mininterval=0, smoothing=0)

        t.update()
        t.update(2)
        t.update(5)  # this should be stored as miniters
        t.update(1)

        out = our_file.getvalue()
        assert all(i in out for i in ("0/10", "1/10", "3/10"))
        assert "2/10" not in out
        assert t.dynamic_miniters and not t.smoothing
        assert t.miniters == 5
        t.close()

    # Check iterable based tldm
    with closing(StringIO()) as our_file:
        t = tldm(range(10), file=our_file, miniters=None, mininterval=0, smoothing=0.5)
        for _ in t:
            pass
        assert t.dynamic_miniters

    # No smoothing
    with closing(StringIO()) as our_file:
        t = tldm(range(10), file=our_file, miniters=None, mininterval=0, smoothing=0)
        for _ in t:
            pass
        assert t.dynamic_miniters

    # No dynamic_miniters (miniters is fixed manually)
    with closing(StringIO()) as our_file:
        t = tldm(range(10), file=our_file, miniters=1, mininterval=0)
        for _ in t:
            pass
        assert not t.dynamic_miniters


def test_big_min_interval() -> None:
    """Test large mininterval"""
    with closing(StringIO()) as our_file:
        for _ in tldm(range(2), file=our_file, mininterval=1e10):
            pass
        assert "50%" not in our_file.getvalue()

    with closing(StringIO()) as our_file, tldm(range(2), file=our_file, mininterval=1e10) as t:
        t.update()
        t.update()
        assert "50%" not in our_file.getvalue()


def test_smoothed_dynamic_min_iters() -> None:
    """Test smoothed dynamic miniters"""
    timer = DiscreteTimer()

    with closing(StringIO()) as our_file:
        with tldm(
            total=100,
            file=our_file,
            miniters=None,
            mininterval=1,
            smoothing=0.5,
            maxinterval=0,
        ) as t:
            cpu_timify(t, timer)

            # Increase 10 iterations at once
            timer.sleep(1)
            t.update(10)
            # The next iterations should be partially skipped
            for _ in range(2):
                timer.sleep(1)
                t.update(4)
            for _ in range(20):
                timer.sleep(1)
                t.update()

            assert t.dynamic_miniters
        out = our_file.getvalue()
    assert "  0%|          | 0/100 [00:00<" in out
    assert "20%" in out
    assert "23%" not in out
    assert "25%" in out
    assert "26%" not in out
    assert "28%" in out


def test_smoothed_dynamic_min_iters_with_min_interval() -> None:
    """Test smoothed dynamic miniters with mininterval"""
    timer = DiscreteTimer()

    # In this test, `miniters` should gradually decline
    total = 100

    with closing(StringIO()) as our_file:
        # Test manual updating tldm
        with tldm(
            total=total,
            file=our_file,
            miniters=None,
            mininterval=1e-3,
            smoothing=1,
            maxinterval=0,
        ) as t:
            cpu_timify(t, timer)

            t.update(10)
            timer.sleep(1e-2)
            for _ in range(4):
                t.update()
                timer.sleep(1e-2)
            out = our_file.getvalue()
            assert t.dynamic_miniters

    with closing(StringIO()) as our_file:
        # Test iteration-based tldm
        with tldm(
            range(total),
            file=our_file,
            miniters=None,
            mininterval=0.01,
            smoothing=1,
            maxinterval=0,
        ) as t2:
            cpu_timify(t2, timer)

            for i in t2:
                if i >= 10:
                    timer.sleep(0.1)
                if i >= 14:
                    break
            out2 = our_file.getvalue()

    assert t.dynamic_miniters
    assert "  0%|          | 0/100 [00:00<" in out
    assert "11%" in out and "11%" in out2
    # assert '12%' not in out and '12%' in out2
    assert "13%" in out and "13%" in out2
    assert "14%" in out and "14%" in out2


@mark.slow
def test_lock_creation(mocker) -> None:
    """Test that importing tldm does not create multiprocessing objects."""

    lock_mock = mocker.patch("multiprocessing.RLock")

    # Importing the module should not create a lock
    from tldm import tldm

    assert lock_mock.call_count == 0
    # Creating a progress bar should use existing lock
    with closing(StringIO()) as our_file, tldm(file=our_file) as _:  # NOQA
        pass

    assert lock_mock.call_count == 0


def test_disable() -> None:
    """Test disable"""
    with closing(StringIO()) as our_file:
        for _ in tldm(range(3), file=our_file, disable=True):
            pass
        assert our_file.getvalue() == ""

    with closing(StringIO()) as our_file:
        progressbar = tldm(total=3, file=our_file, miniters=1, disable=True)
        progressbar.update(3)
        progressbar.close()
        assert our_file.getvalue() == ""


def test_complete_bar_on_early_finish() -> None:
    """Test completing the bar on early exit."""
    with closing(StringIO()) as our_file:
        with tldm(
            range(10),
            total=10,
            file=our_file,
            miniters=1,
            mininterval=0,
            complete_bar_on_early_finish=True,
        ) as pbar:
            for i in pbar:
                if i == 4:
                    break
        assert "10/10" in our_file.getvalue()


def test_no_complete_bar_on_exception() -> None:
    """Test not completing the bar on exception."""
    with closing(StringIO()) as our_file:
        try:
            with tldm(
                range(10),
                total=10,
                file=our_file,
                miniters=1,
                mininterval=0,
                complete_bar_on_early_finish=True,
            ) as pbar:
                for i in pbar:
                    if i == 4:
                        raise RuntimeError("boom")
        except RuntimeError:
            pass
        out = our_file.getvalue()
        assert "10/10" not in out


def test_nototal() -> None:
    """Test unknown total length"""

    def unknown_length_run():
        yield from range(10)

    with closing(StringIO()) as our_file:
        for _ in tldm(iter(unknown_length_run()), file=our_file, unit_scale=10):
            pass

        assert "100it" in our_file.getvalue()

    # TODO this is printing 0 as the total number of iters, but should be a ?
    with closing(StringIO()) as our_file:
        for _ in tldm(iter(unknown_length_run()), file=our_file, bar_format="{l_bar}{bar}{r_bar}"):
            pass
        print(our_file.getvalue())
        assert "10/?" in our_file.getvalue()


def test_unit() -> None:
    """Test SI unit prefix"""
    with closing(StringIO()) as our_file:
        for _ in tldm(range(3), file=our_file, miniters=1, unit="bytes"):
            pass
        assert "bytes/s" in our_file.getvalue()


def test_ascii() -> None:
    """Test ascii/unicode bar"""
    # Test ascii autodetection
    with closing(StringIO()) as our_file, tldm(total=10, file=our_file, ascii=None) as t:
        assert t.ascii  # TODO: this may fail in the future

    # Test ascii bar
    with closing(StringIO()) as our_file:
        for _ in tldm(range(3), total=15, file=our_file, miniters=1, mininterval=0, ascii=True):
            pass
        res = our_file.getvalue().strip("\r").split("\r")
    assert "7%|6" in res[1]
    assert "13%|#3" in res[2]
    assert "20%|##" in res[3]

    # Test unicode bar
    with closing(UnicodeIO()) as our_file:
        with tldm(total=15, file=our_file, ascii=False, mininterval=0) as t:
            for _ in range(3):
                t.update()
        res = our_file.getvalue().strip("\r").split("\r")
    assert "7%|\u258b" in res[1]
    assert "13%|\u2588\u258e" in res[2]
    assert "20%|\u2588\u2588" in res[3]

    # Test custom bar
    for bars in [" .oO0", " #"]:
        with closing(StringIO()) as our_file:
            for _ in tldm(
                range(len(bars) - 1),
                file=our_file,
                miniters=1,
                mininterval=0,
                ascii=bars,
                ncols=27,
            ):
                pass
            res = our_file.getvalue().strip("\r").split("\r")
        for b, line in zip(bars, res):
            assert "|" + b + "|" in line


def test_update() -> None:
    """Test manual creation and updates"""
    res = None
    with closing(StringIO()) as our_file:
        with tldm(total=2, file=our_file, miniters=1, mininterval=0) as progressbar:
            assert len(progressbar) == 2
            progressbar.update(2)
            assert "| 2/2" in our_file.getvalue()
            progressbar.desc = "dynamically notify of 4 increments in total"
            progressbar.total = 4
            progressbar.update(-1)
            progressbar.update(2)
        res = our_file.getvalue()
    assert "| 3/4 " in res
    assert "dynamically notify of 4 increments in total" in res


def test_close() -> None:
    """Test manual creation and closure and n_instances"""

    # With `leave` option
    with closing(StringIO()) as our_file:
        progressbar = tldm(total=3, file=our_file, miniters=10)
        progressbar.update(3)
        assert "| 3/3 " not in our_file.getvalue()  # Should be blank
        assert len(tldm._instances) == 1
        progressbar.close()
        assert len(tldm._instances) == 0
        assert "| 3/3 " in our_file.getvalue()

    # Without `leave` option
    with closing(StringIO()) as our_file:
        progressbar = tldm(total=3, file=our_file, miniters=10, leave=False)
        progressbar.update(3)
        progressbar.close()
        assert "| 3/3 " not in our_file.getvalue()  # Should be blank

    # With all updates
    with closing(StringIO()) as our_file:
        assert len(tldm._instances) == 0
        with tldm(total=3, file=our_file, miniters=0, mininterval=0, leave=True) as progressbar:
            assert len(tldm._instances) == 1
            progressbar.update(3)
            res = our_file.getvalue()
            assert "| 3/3 " in res  # Should be blank
            assert "\n" not in res
        # close() called
        assert len(tldm._instances) == 0

        exres = res.rsplit(", ", 1)[0]
        res = our_file.getvalue()
        assert res[-1] == "\n"
        if not res.startswith(exres):
            raise AssertionError(f"\n<<< Expected:\n{exres}, ...it/s]\n>>> Got:\n{res}\n===")

    # Closing after the output stream has closed
    with closing(StringIO()) as our_file:
        t = tldm(total=2, file=our_file)
        t.update()
        t.update()
    t.close()


def test_smoothing() -> None:
    """Test exponential weighted average smoothing"""
    timer = DiscreteTimer()

    # -- Test disabling smoothing
    with closing(StringIO()) as our_file:
        with tldm(range(3), file=our_file, smoothing=0, leave=True) as t:
            cpu_timify(t, timer)

            for _ in t:
                pass
        assert "| 3/3 " in our_file.getvalue()

    # -- Test smoothing
    # 1st case: no smoothing (only use average)
    with closing(StringIO()) as our_file2:
        with closing(StringIO()) as our_file:
            t = tldm(
                range(3),
                file=our_file2,
                smoothing=0,
                leave=True,
                miniters=1,
                mininterval=0,
            )
            cpu_timify(t, timer)

            with tldm(
                range(3),
                file=our_file,
                smoothing=0,
                leave=True,
                miniters=1,
                mininterval=0,
            ) as t2:
                cpu_timify(t2, timer)

                for i in t2:
                    # Sleep more for first iteration and
                    # see how quickly rate is updated
                    if i == 0:
                        timer.sleep(0.01)
                    else:
                        # Need to sleep in all iterations
                        # to calculate smoothed rate
                        # (else delta_t is 0!)
                        timer.sleep(0.001)
                    t.update()
            n_old = len(tldm._instances)
            t.close()
            assert len(tldm._instances) == n_old - 1
            # Get result for iter-based bar
            a = progressbar_rate(get_bar(our_file.getvalue(), 3))
        # Get result for manually updated bar
        a2 = progressbar_rate(get_bar(our_file2.getvalue(), 3))

    # 2nd case: use max smoothing (= instant rate)
    with closing(StringIO()) as our_file2:
        with closing(StringIO()) as our_file:
            t = tldm(
                range(3),
                file=our_file2,
                smoothing=1,
                leave=True,
                miniters=1,
                mininterval=0,
            )
            cpu_timify(t, timer)

            with tldm(
                range(3),
                file=our_file,
                smoothing=1,
                leave=True,
                miniters=1,
                mininterval=0,
            ) as t2:
                cpu_timify(t2, timer)

                for i in t2:
                    if i == 0:
                        timer.sleep(0.01)
                    else:
                        timer.sleep(0.001)
                    t.update()
            t.close()
            # Get result for iter-based bar
            b = progressbar_rate(get_bar(our_file.getvalue(), 3))
        # Get result for manually updated bar
        b2 = progressbar_rate(get_bar(our_file2.getvalue(), 3))

    # 3rd case: use medium smoothing
    with closing(StringIO()) as our_file2:
        with closing(StringIO()) as our_file:
            t = tldm(
                range(3),
                file=our_file2,
                smoothing=0.5,
                leave=True,
                miniters=1,
                mininterval=0,
            )
            cpu_timify(t, timer)

            t2 = tldm(
                range(3),
                file=our_file,
                smoothing=0.5,
                leave=True,
                miniters=1,
                mininterval=0,
            )
            cpu_timify(t2, timer)

            for i in t2:
                if i == 0:
                    timer.sleep(0.01)
                else:
                    timer.sleep(0.001)
                t.update()
            t2.close()
            t.close()
            # Get result for iter-based bar
            c = progressbar_rate(get_bar(our_file.getvalue(), 3))
        # Get result for manually updated bar
        c2 = progressbar_rate(get_bar(our_file2.getvalue(), 3))

    # Check that medium smoothing's rate is between no and max smoothing rates
    assert a <= c <= b
    assert a2 <= c2 <= b2


def test_bar_format() -> None:
    """Test custom bar formatting"""
    with closing(StringIO()) as our_file:
        bar_format = (
            "{l_bar}{bar}|{n_fmt}/{total_fmt}-{n}/{total}"
            "{percentage}{rate}{rate_fmt}{elapsed}{remaining}"
        )
        for _ in trange(2, file=our_file, leave=True, bar_format=bar_format):
            pass
        out = our_file.getvalue()
    assert "\r  0%|          |0/2-0/20.0None?it/s00:00?\r" in out

    # Test unicode string auto conversion
    with closing(StringIO()) as our_file:
        bar_format = r"hello world"
        with tldm(ascii=False, bar_format=bar_format, file=our_file) as t:
            assert isinstance(t.bar_format, str)


def test_custom_format() -> None:
    """Test adding additional derived format arguments"""

    class TldmExtraFormat(tldm):
        """Provides a `total_time` format parameter"""

        @property
        def format_dict(self):
            d = super().format_dict
            total_time = d["elapsed"] * (d["total"] or 0) / max(d["n"], 1)
            d.update(total_time=format_interval(total_time) + " in total")
            return d

    with closing(StringIO()) as our_file:
        for _ in TldmExtraFormat(
            range(10),
            file=our_file,
            bar_format="{total_time}: {percentage:.0f}%|{bar}{r_bar}",
        ):
            pass
        assert "00:00 in total" in our_file.getvalue()


def test_eta(capsys):
    """Test eta bar_format"""
    from datetime import datetime as dt

    for _ in trange(
        999, miniters=1, mininterval=0, leave=True, bar_format="{l_bar}{eta:%Y-%m-%d}"
    ):
        pass
    _, err = capsys.readouterr()
    assert f"\r100%|{dt.now():%Y-%m-%d}\n" in err


def test_unpause() -> None:
    """
    Test unpause
    """
    timer = DiscreteTimer()
    with closing(StringIO()) as our_file:
        t = trange(10, file=our_file, leave=True, mininterval=0)
        cpu_timify(t, timer)
        timer.sleep(0.01)
        t.update()
        timer.sleep(0.01)
        t.update()
        t.pause()
        timer.sleep(0.1)  # longer wait time
        t.unpause()
        timer.sleep(0.01)
        t.update()
        timer.sleep(0.01)
        t.update()
        t.close()
        r_before = progressbar_rate(get_bar(our_file.getvalue(), 2))
        r_after = progressbar_rate(get_bar(our_file.getvalue(), 3))
    assert r_before == r_after


def test_disabled_unpause(capsys):
    """Test disabled unpause"""
    with tldm(total=10, disable=True) as t:
        t.update()
        t.unpause()
        t.update()
        print(t)
    out, err = capsys.readouterr()
    assert not err
    assert out == "  0%|          | 0/10 [00:00<?, ?it/s]\n"


def test_reset() -> None:
    """Test resetting a bar for re-use"""
    with closing(StringIO()) as our_file:
        with tldm(total=10, file=our_file, miniters=1, mininterval=0, maxinterval=0) as t:
            t.update(9)
            t.reset()
            t.update()
            t.reset(total=12)
            t.update(10)
        assert "| 1/10" in our_file.getvalue()
        assert "| 10/12" in our_file.getvalue()


def test_disabled_reset(capsys):
    """Test disabled reset"""
    with tldm(total=10, disable=True) as t:
        t.update(9)
        t.reset()
        t.update()
        t.reset(total=12)
        t.update(10)
        print(t)
    out, err = capsys.readouterr()
    assert not err
    assert out == "  0%|          | 0/12 [00:00<?, ?it/s]\n"


@mark.skipif(nt_and_no_colorama, reason="Windows without colorama")
def test_position() -> None:
    """Test positioned progress bars"""
    # Artificially test nested loop printing
    # Without leave
    our_file = StringIO()
    kwargs = {"file": our_file, "miniters": 1, "mininterval": 0, "maxinterval": 0}
    t = tldm(total=2, desc="pos2 bar", leave=False, position=2, **kwargs)
    t.update()
    t.close()
    out = our_file.getvalue()
    res = [m[0] for m in RE_pos.findall(out)]
    exres = ["\n\n\rpos2 bar:   0%", "\n\n\rpos2 bar:  50%", "\n\n\r      "]

    pos_line_diff(res, exres)

    # Test iteration-based tldm positioning
    our_file = StringIO()
    kwargs["file"] = our_file
    for _ in trange(2, desc="pos0 bar", position=0, **kwargs):
        for _ in trange(2, desc="pos1 bar", position=1, **kwargs):
            for _ in trange(2, desc="pos2 bar", position=2, **kwargs):
                pass
    out = our_file.getvalue()
    res = [m[0] for m in RE_pos.findall(out)]
    exres = [
        "\rpos0 bar:   0%",
        "\n\rpos1 bar:   0%",
        "\n\n\rpos2 bar:   0%",
        "\n\n\rpos2 bar:  50%",
        "\n\n\rpos2 bar: 100%",
        "\rpos2 bar: 100%",
        "\n\n\rpos1 bar:  50%",
        "\n\n\rpos2 bar:   0%",
        "\n\n\rpos2 bar:  50%",
        "\n\n\rpos2 bar: 100%",
        "\rpos2 bar: 100%",
        "\n\n\rpos1 bar: 100%",
        "\rpos1 bar: 100%",
        "\n\rpos0 bar:  50%",
        "\n\rpos1 bar:   0%",
        "\n\n\rpos2 bar:   0%",
        "\n\n\rpos2 bar:  50%",
        "\n\n\rpos2 bar: 100%",
        "\rpos2 bar: 100%",
        "\n\n\rpos1 bar:  50%",
        "\n\n\rpos2 bar:   0%",
        "\n\n\rpos2 bar:  50%",
        "\n\n\rpos2 bar: 100%",
        "\rpos2 bar: 100%",
        "\n\n\rpos1 bar: 100%",
        "\rpos1 bar: 100%",
        "\n\rpos0 bar: 100%",
        "\rpos0 bar: 100%",
        "\n",
    ]
    pos_line_diff(res, exres)

    # Test manual tldm positioning
    our_file = StringIO()
    kwargs["file"] = our_file
    kwargs["total"] = 2
    t1 = tldm(desc="pos0 bar", position=0, **kwargs)
    t2 = tldm(desc="pos1 bar", position=1, **kwargs)
    t3 = tldm(desc="pos2 bar", position=2, **kwargs)
    for _ in range(2):
        t1.update()
        t3.update()
        t2.update()
    out = our_file.getvalue()
    res = [m[0] for m in RE_pos.findall(out)]
    exres = [
        "\rpos0 bar:   0%",
        "\n\rpos1 bar:   0%",
        "\n\n\rpos2 bar:   0%",
        "\rpos0 bar:  50%",
        "\n\n\rpos2 bar:  50%",
        "\n\rpos1 bar:  50%",
        "\rpos0 bar: 100%",
        "\n\n\rpos2 bar: 100%",
        "\n\rpos1 bar: 100%",
    ]
    pos_line_diff(res, exres)
    t1.close()
    t2.close()
    t3.close()

    # Test auto repositioning of bars when a bar is prematurely closed
    # tldm._instances.clear()  # reset number of instances
    with closing(StringIO()) as our_file:
        t1 = tldm(total=10, file=our_file, desc="1.pos0 bar", mininterval=0)
        t2 = tldm(total=10, file=our_file, desc="2.pos1 bar", mininterval=0)
        t3 = tldm(total=10, file=our_file, desc="3.pos2 bar", mininterval=0)
        res = [m[0] for m in RE_pos.findall(our_file.getvalue())]
        exres = ["\r1.pos0 bar:   0%", "\n\r2.pos1 bar:   0%", "\n\n\r3.pos2 bar:   0%"]
        pos_line_diff(res, exres)

        t2.close()
        t4 = tldm(total=10, file=our_file, desc="4.pos2 bar", mininterval=0)
        t1.update(1)
        t3.update(1)
        t4.update(1)
        res = [m[0] for m in RE_pos.findall(our_file.getvalue())]
        exres = [
            *exres,
            "\r2.pos1 bar:   0%",
            "\n\n\r3.pos2 bar:   0%",
            "\n\n\r4.pos2 bar:   0%",
            "\r1.pos0 bar:  10%",
            "\n\r3.pos2 bar:  10%",
            "\n\n\r4.pos2 bar:  10%",
        ]
        pos_line_diff(res, exres)
        t4.close()
        t3.close()
        t1.close()


@mark.skipif(nt_and_no_colorama, reason="Windows without colorama")
@mark.parametrize(
    "leave",
    [
        True,  # all bars remain
        False,  # no bars remain
        None,  # only first bar remains
    ],
)
def test_position_leave(leave: bool):
    """Test leaving of nested positioned progress bars"""
    our_file = StringIO()
    kwargs = {
        "file": our_file,
        "miniters": 1,
        "mininterval": 0,
        "maxinterval": 0,
        "leave": leave,
        "bar_format": "{desc}: {percentage:3.0f}%",
    }
    for _ in trange(2, desc="pos0 bar", position=0, **kwargs):
        t2 = tldm(total=2, desc="pos1 bar", position=1, **kwargs)
        t2.update()
        t3 = tldm(total=2, desc="pos2 bar", position=2, **kwargs)
        t3.update()
        # complete t2 before t3
        t2.update()
        t2.close()
        t3.update()
        t3.close()

    out = our_file.getvalue()
    res = [m[0] for m in RE_pos.findall(out)]
    # Bar 2 being left from the screen means bar 3 needs extra newline when
    # positioning. If it is not left, then bar 3 needs to be cleared in its old
    # position and redrawn in gap left by bar 2.
    if leave:
        bar2left, bar3move = "\n", []
    else:
        bar2left, bar3move = "", ["\n\n\r            ", "\r\x1b[A\x1b[A"]
    innerex = [
        "\n\rpos1 bar:   0%",
        "\n\rpos1 bar:  50%",
        "\n\n\rpos2 bar:   0%",
        "\n\n\rpos2 bar:  50%",
        "\n\rpos1 bar: 100%",
        "\rpos1 bar: 100%" if leave else "\n\r             ",
        *bar3move,
        bar2left + "\n\rpos2 bar:  50%",
        "\n\rpos2 bar: 100%",
        "\rpos2 bar: 100%" if leave else "\n\r             ",
    ]
    # Bar 1 being left on screen adds an extra newline to the output
    # that then shows up as part of the next res line.
    bar1left = "\n" if leave else ""
    exres = [
        "\rpos0 bar:   0%",
        *innerex,
        bar1left + "\rpos0 bar:  50%",
        *innerex,
        bar1left + "\rpos0 bar: 100%",
        "\rpos0 bar: 100%" if leave is not False else "\r              ",
        "\n" if leave is not False else "\r",
    ]
    pos_line_diff(res, exres)


def test_set_description() -> None:
    """Test set description"""
    with closing(StringIO()) as our_file:
        with tldm(desc="Hello", file=our_file) as t:
            assert t.desc == "Hello"
            t.set_description_str("World")
            assert t.desc == "World"
            t.set_description()
            assert t.desc == ""
            t.set_description("Bye")
            assert t.desc == "Bye: "
        assert "World" in our_file.getvalue()

    # without refresh
    with closing(StringIO()) as our_file:
        with tldm(desc="Hello", file=our_file) as t:
            assert t.desc == "Hello"
            t.set_description_str("World", False)
            assert t.desc == "World"
            t.set_description(None, False)
            assert t.desc == ""
        assert "World" not in our_file.getvalue()

    # unicode
    with closing(StringIO()) as our_file, tldm(total=10, file=our_file) as t:
        t.set_description("\xe1\xe9\xed\xf3\xfa")


def test_cmp(capsys):
    """Test comparison functions"""
    t0 = tldm(total=10)
    t1 = tldm(total=10)
    t2 = tldm(total=10)

    assert t0 < t1
    assert t2 >= t0
    assert t0 <= t2

    t3 = tldm(total=10)
    t4 = tldm(total=10)
    t5 = tldm(total=10)
    t5.close()
    t6 = tldm(total=10)

    assert t3 != t4
    assert t3 > t2
    assert t5 == t6
    t6.close()
    t4.close()
    t3.close()
    t2.close()
    t1.close()
    t0.close()
    out, err = capsys.readouterr()
    assert not out
    assert " 0/10 " in err


# https://docs.python.org/3/tutorial/datastructures.html#comparing-sequences-and-other-types
@mark.parametrize(
    "left,right",
    [
        ((1, 2, 3), (1, 2, 4)),
        ([1, 2, 3], [1, 2, 4]),
        ("ABC", "C"),
        ("C", "Pascal"),
        ("Pascal", "Python"),
        ((1, 2, 3, 4), (1, 2, 4)),
        ((1, 2), (1, 2, -1)),
        ((1, 2, ("aa", "ab")), (1, 2, ("abc", "a"), 4)),
    ],
)
def test_cmp_iterables(capsys, left, right):
    """Test iterable comparison"""
    assert (left < right) == (tldm(left) < right)
    assert (left == right) == (tldm(left) == right)
    assert (left > right) == (tldm(left) > right)
    out, err = capsys.readouterr()
    assert not out
    assert f"/{len(left):d} " in err


def test_repr() -> None:
    """Test representation"""
    with closing(StringIO()) as our_file, tldm(total=10, ascii=True, file=our_file) as t:
        assert str(t) == "  0%|          | 0/10 [00:00<?, ?it/s]"


def test_clear() -> None:
    """Test clearing bar display"""
    with closing(StringIO()) as our_file:
        t1 = tldm(total=10, file=our_file, desc="pos0 bar", bar_format="{l_bar}")
        t2 = trange(10, file=our_file, desc="pos1 bar", bar_format="{l_bar}")
        before = squash_ctrlchars(our_file.getvalue())
        t2.clear()
        t1.clear()
        after = squash_ctrlchars(our_file.getvalue())
        t1.close()
        t2.close()
        assert before == ["pos0 bar:   0%|", "pos1 bar:   0%|"]
        assert after == ["", ""]


def test_clear_disabled() -> None:
    """Test disabled clear"""
    with closing(StringIO()) as our_file:
        with tldm(
            total=10, file=our_file, desc="pos0 bar", disable=True, bar_format="{l_bar}"
        ) as t:
            t.clear()
        assert our_file.getvalue() == ""


def test_refresh() -> None:
    """Test refresh bar display"""
    with closing(StringIO()) as our_file:
        t1 = tldm(
            total=10,
            file=our_file,
            desc="pos0 bar",
            bar_format="{l_bar}",
            mininterval=999,
            miniters=999,
        )
        t2 = tldm(
            total=10,
            file=our_file,
            desc="pos1 bar",
            bar_format="{l_bar}",
            mininterval=999,
            miniters=999,
        )
        t1.update()
        t2.update()
        before = squash_ctrlchars(our_file.getvalue())
        t1.refresh()
        t2.refresh()
        after = squash_ctrlchars(our_file.getvalue())
        t1.close()
        t2.close()

        # Check that refreshing indeed forced the display to use realtime state
        assert before == ["pos0 bar:   0%|", "pos1 bar:   0%|"]
        assert after == ["pos0 bar:  10%|", "pos1 bar:  10%|"]


def test_disabled_repr(capsys):
    """Test disabled repr"""
    with tldm(total=10, disable=True) as t:
        str(t)
        t.update()
        print(t)
    out, err = capsys.readouterr()
    assert not err
    assert out == "  0%|          | 0/10 [00:00<?, ?it/s]\n"


def test_disabled_refresh() -> None:
    """Test disabled refresh"""
    with closing(StringIO()) as our_file:
        with tldm(
            total=10,
            file=our_file,
            desc="pos0 bar",
            disable=True,
            bar_format="{l_bar}",
            mininterval=999,
            miniters=999,
        ) as t:
            t.update()
            t.refresh()

        assert our_file.getvalue() == ""


def test_write() -> None:
    """Test write messages"""
    s = "Hello world"
    with closing(StringIO()) as our_file:
        # Change format to keep only left part w/o bar and it/s rate
        t1 = tldm(
            total=10,
            file=our_file,
            desc="pos0 bar",
            bar_format="{l_bar}",
            mininterval=0,
            miniters=1,
        )
        t2 = trange(
            10,
            file=our_file,
            desc="pos1 bar",
            bar_format="{l_bar}",
            mininterval=0,
            miniters=1,
        )
        t3 = tldm(
            total=10,
            file=our_file,
            desc="pos2 bar",
            bar_format="{l_bar}",
            mininterval=0,
            miniters=1,
        )
        t1.update()
        t2.update()
        t3.update()
        before = our_file.getvalue()

        # Write msg and see if bars are correctly redrawn below the msg
        t1.write(s, file=our_file)  # call as an instance method
        tldm.write(s, file=our_file)  # call as a class method
        after = our_file.getvalue()

        t1.close()
        t2.close()
        t3.close()

        before_squashed = squash_ctrlchars(before)
        after_squashed = squash_ctrlchars(after)

        assert after_squashed == [s, s] + before_squashed

    # Check that no bar clearing if different file
    with closing(StringIO()) as our_file_bar, closing(StringIO()) as our_file_write:
        t1 = tldm(
            total=10,
            file=our_file_bar,
            desc="pos0 bar",
            bar_format="{l_bar}",
            mininterval=0,
            miniters=1,
        )

        t1.update()
        before_bar = our_file_bar.getvalue()

        tldm.write(s, file=our_file_write)

        after_bar = our_file_bar.getvalue()
        t1.close()

        assert before_bar == after_bar

    # Test stdout/stderr anti-mixup strategy
    # Backup stdout/stderr
    stde = sys.stderr
    stdo = sys.stdout
    # Mock stdout/stderr
    with closing(StringIO()) as our_stderr, closing(StringIO()) as our_stdout:
        sys.stderr = our_stderr
        sys.stdout = our_stdout
        t1 = tldm(
            total=10,
            file=sys.stderr,
            desc="pos0 bar",
            bar_format="{l_bar}",
            mininterval=0,
            miniters=1,
        )

        t1.update()
        before_err = sys.stderr.getvalue()
        before_out = sys.stdout.getvalue()

        tldm.write(s, file=sys.stdout)
        after_err = sys.stderr.getvalue()
        after_out = sys.stdout.getvalue()

        t1.close()

        assert before_err == "\rpos0 bar:   0%|\rpos0 bar:  10%|"
        assert before_out == ""
        after_err_res = [m[0] for m in RE_pos.findall(after_err)]
        exres = [
            "\rpos0 bar:   0%|",
            "\rpos0 bar:  10%|",
            "\r               ",
            "\r\rpos0 bar:  10%|",
        ]
        pos_line_diff(after_err_res, exres)
        assert after_out == s + "\n"
    # Restore stdout and stderr
    sys.stderr = stde
    sys.stdout = stdo


def test_print() -> None:
    """Test print values"""
    values = ["Hello", "world", 123, 3.141592653589793, set("Python"), dict]
    with closing(StringIO()) as our_file:
        # Change format to keep only left part w/o bar and it/s rate
        t1 = tldm(
            total=10,
            file=our_file,
            desc="pos0 bar",
            bar_format="{l_bar}",
            mininterval=0,
            miniters=1,
        )
        t2 = trange(
            10,
            file=our_file,
            desc="pos1 bar",
            bar_format="{l_bar}",
            mininterval=0,
            miniters=1,
        )
        t3 = tldm(
            total=10,
            file=our_file,
            desc="pos2 bar",
            bar_format="{l_bar}",
            mininterval=0,
            miniters=1,
        )
        t1.update()
        t2.update()
        t3.update()
        before = our_file.getvalue()

        # Write msg and see if bars are correctly redrawn below the msg
        t1.print(*values, file=our_file)  # call as an instance method
        tldm.print(*values, file=our_file)  # call as a class method
        after = our_file.getvalue()

        t1.close()
        t2.close()
        t3.close()

        before_squashed = squash_ctrlchars(before)
        after_squashed = squash_ctrlchars(after)

        assert after_squashed == [" ".join(f"{v}" for v in values)] * 2 + before_squashed

    # Check that no bar clearing if different file
    with closing(StringIO()) as our_file_bar, closing(StringIO()) as our_file_write:
        t1 = tldm(
            total=10,
            file=our_file_bar,
            desc="pos0 bar",
            bar_format="{l_bar}",
            mininterval=0,
            miniters=1,
        )

        t1.update()
        before_bar = our_file_bar.getvalue()

        tldm.print(*values, file=our_file_write)

        after_bar = our_file_bar.getvalue()
        t1.close()

        assert before_bar == after_bar

    # Test stdout/stderr anti-mixup strategy
    # Backup stdout/stderr
    stde = sys.stderr
    stdo = sys.stdout
    # Mock stdout/stderr
    with closing(StringIO()) as our_stderr, closing(StringIO()) as our_stdout:
        sys.stderr = our_stderr
        sys.stdout = our_stdout
        t1 = tldm(
            total=10,
            file=sys.stderr,
            desc="pos0 bar",
            bar_format="{l_bar}",
            mininterval=0,
            miniters=1,
        )

        t1.update()
        before_err = sys.stderr.getvalue()
        before_out = sys.stdout.getvalue()

        tldm.print(*values, file=sys.stdout)
        after_err = sys.stderr.getvalue()
        after_out = sys.stdout.getvalue()

        t1.close()

        assert before_err == "\rpos0 bar:   0%|\rpos0 bar:  10%|"
        assert before_out == ""
        after_err_res = [m[0] for m in RE_pos.findall(after_err)]
        exres = [
            "\rpos0 bar:   0%|",
            "\rpos0 bar:  10%|",
            "\r               ",
            "\r\rpos0 bar:  10%|",
        ]
        pos_line_diff(after_err_res, exres)
        assert after_out == " ".join(f"{v}" for v in values) + "\n"
    # Restore stdout and stderr
    sys.stderr = stde
    sys.stdout = stdo


def test_len() -> None:
    """Test advance len (numpy array shape)"""
    np = importorskip("numpy")
    with closing(StringIO()) as f, tldm(np.zeros((3, 4)), file=f) as t:
        assert len(t) == 3


def test_autodisable_disable() -> None:
    """Test autodisable will disable on non-TTY"""
    with closing(StringIO()) as our_file:
        with tldm(total=10, disable=True, file=our_file) as t:
            t.update(3)
        assert our_file.getvalue() == ""


def test_autodisable_enable() -> None:
    """Test autodisable will not disable on TTY"""
    with closing(StringIO()) as our_file:
        with tldm(total=10, disable=False, file=our_file) as t:
            t.update()
        assert our_file.getvalue() != ""


def test_postfix() -> None:
    """Test postfix"""
    postfix = {"float": 0.321034, "gen": 543, "str": "h", "lst": [2]}
    postfix_order = (("w", "w"), ("a", 0))  # no need for OrderedDict
    expected = ["float=0.321", "gen=543", "lst=[2]", "str=h"]
    expected_order = ["w=w", "a=0", "float=0.321", "gen=543", "lst=[2]", "str=h"]

    # Test postfix set at init
    with (
        closing(StringIO()) as our_file,
        tldm(
            total=10,
            file=our_file,
            desc="pos0 bar",
            bar_format="{r_bar}",
            postfix=postfix,
        ) as t1,
    ):
        t1.refresh()
        out = our_file.getvalue()

    # Test postfix set after init
    with (
        closing(StringIO()) as our_file,
        trange(10, file=our_file, desc="pos1 bar", bar_format="{r_bar}", postfix=None) as t2,
    ):
        t2.set_postfix(**postfix)
        t2.refresh()
        out2 = our_file.getvalue()

    # Order of items in dict may change, so need a loop to check per item
    for res in expected:
        assert res in out
        assert res in out2

    # Test postfix (with ordered dict and no refresh) set after init
    with (
        closing(StringIO()) as our_file,
        trange(10, file=our_file, desc="pos2 bar", bar_format="{r_bar}", postfix=None) as t3,
    ):
        t3.set_postfix(postfix_order, False, **postfix)
        t3.refresh()  # explicit external refresh
        out3 = our_file.getvalue()

    out3 = out3[1:-1].split(", ")[3:]
    assert out3 == expected_order

    # Test postfix (with ordered dict and refresh) set after init
    with (
        closing(StringIO()) as our_file,
        trange(10, file=our_file, desc="pos2 bar", bar_format="{r_bar}", postfix=None) as t4,
    ):
        t4.set_postfix(postfix_order, True, **postfix)
        t4.refresh()  # double refresh
        out4 = our_file.getvalue()

    assert out4.count("\r") > out3.count("\r")
    assert out4.count(", ".join(expected_order)) == 2

    # Test setting postfix string directly
    with (
        closing(StringIO()) as our_file,
        trange(10, file=our_file, desc="pos2 bar", bar_format="{r_bar}", postfix=None) as t5,
    ):
        t5.set_postfix_str("Hello", False)
        t5.set_postfix_str("World")
        out5 = our_file.getvalue()

    assert "Hello" not in out5
    out5 = out5[1:-1].split(", ")[3:]
    assert out5 == ["World"]


def test_postfix_direct() -> None:
    """Test directly assigning non-str objects to postfix"""
    with closing(StringIO()) as our_file:
        with tldm(
            total=10,
            file=our_file,
            miniters=1,
            mininterval=0,
            bar_format="{postfix[0][name]} {postfix[1]:>5.2f}",
            postfix=[{"name": "foo"}, 42],
        ) as t:
            for i in range(10):
                if i % 2:
                    t.postfix[0]["name"] = "abcdefghij"[i]
                else:
                    t.postfix[1] = i
                t.update()
        res = our_file.getvalue()
        assert "f  6.00" in res
        assert "h  6.00" in res
        assert "h  8.00" in res
        assert "j  8.00" in res


@contextmanager
def std_out_err_redirect_tldm(tldm_file=sys.stderr):
    orig_out_err = sys.stdout, sys.stderr
    try:
        sys.stdout = sys.stderr = DummyTldmFile(tldm_file)
        yield orig_out_err[0]
    # Relay exceptions
    except Exception as exc:
        raise exc
    # Always restore sys.stdout/err if necessary
    finally:
        sys.stdout, sys.stderr = orig_out_err


def test_file_redirection() -> None:
    """Test redirection of output"""
    with closing(StringIO()) as our_file:
        # Redirect stdout to tldm.write()
        with std_out_err_redirect_tldm(tldm_file=our_file), tldm(total=3) as pbar:
            print("Such fun")
            pbar.update(1)
            print("Such", "fun")
            pbar.update(1)
            print("Such ", end="")
            print("fun")
            pbar.update(1)
        res = our_file.getvalue()
        assert res.count("Such fun\n") == 3
        assert "0/3" in res
        assert "3/3" in res


def test_external_write() -> None:
    """Test external write mode"""
    with closing(StringIO()) as our_file:
        # Redirect stdout to tldm.write()
        for _ in trange(3, file=our_file):
            del tldm._lock  # classmethod should be able to recreate lock
            with tldm.external_write_mode(file=our_file):
                our_file.write("Such fun\n")
        res = our_file.getvalue()
        assert res.count("Such fun\n") == 3
        assert "0/3" in res
        assert "3/3" in res


def test_unit_scale() -> None:
    """Test numeric `unit_scale`"""
    with closing(StringIO()) as our_file:
        for _ in tldm(range(9), unit_scale=9, file=our_file, miniters=1, mininterval=0):
            pass
        out = our_file.getvalue()
        assert "81/81" in out


@patch_lock(thread=False)
def test_threading() -> None:
    """Test multiprocess/thread-realted features"""
    pass  # TODO: test interleaved output #445


def test_bool() -> None:
    """Test boolean cast"""

    def internal(our_file, disable):
        kwargs = {"file": our_file, "disable": disable}
        with trange(10, **kwargs) as t:
            assert t
        with trange(0, **kwargs) as t:
            assert not t
        with tldm(total=10, **kwargs) as t:
            assert bool(t)
        with tldm(total=0, **kwargs) as t:
            assert not bool(t)
        with tldm([], **kwargs) as t:
            assert not t
        with tldm([0], **kwargs) as t:
            assert t
        with tldm(iter([]), **kwargs) as t:
            assert t
        with tldm(iter([1, 2, 3]), **kwargs) as t:
            assert t
        with tldm(**kwargs) as t:
            try:
                print(bool(t))
            except TypeError:
                pass
            else:
                raise TypeError("Expected bool(tldm()) to fail")

    # test with and without disable
    with closing(StringIO()) as our_file:
        internal(our_file, False)
        internal(our_file, True)


def test_wrapattr() -> None:
    """Test wrapping file-like objects"""
    data = "a twenty-char string"

    with closing(StringIO()) as our_file:
        with closing(StringIO()) as writer:
            with tldm.wrapattr(writer, "write", file=our_file, bytes=True) as wrap:
                wrap.write(data)
            res = writer.getvalue()
            assert data == res
        res = our_file.getvalue()
        assert "%.1fB [" % len(data) in res

    with closing(StringIO()) as our_file:
        with closing(StringIO()) as writer:
            with tldm.wrapattr(writer, "write", file=our_file, bytes=False) as wrap:
                wrap.write(data)
        res = our_file.getvalue()
        assert "%dit [" % len(data) in res


def test_float_progress() -> None:
    """Test float totals"""
    with closing(StringIO()) as our_file, trange(10, total=9.6, file=our_file) as t:
        with catch_warnings(record=True) as w:
            simplefilter("always", category=TldmWarning)
            for i in t:
                if i < 9:
                    assert not w
            assert w
            assert "clamping frac" in str(w[-1].message)


def test_screen_shape() -> None:
    """Test screen shape"""
    # ncols
    with closing(StringIO()) as our_file:
        with trange(10, file=our_file, ncols=50) as t:
            list(t)

        res = our_file.getvalue()
        assert all(len(i) == 50 for i in get_bar(res))

    # no second/third bar, leave=False
    with closing(StringIO()) as our_file:
        kwargs = {
            "file": our_file,
            "ncols": 50,
            "nrows": 2,
            "miniters": 0,
            "mininterval": 0,
            "leave": False,
        }
        with trange(10, desc="one", **kwargs) as t1:
            with trange(10, desc="two", **kwargs) as t2:
                with trange(10, desc="three", **kwargs) as t3:
                    list(t3)
                list(t2)
            list(t1)

        res = our_file.getvalue()
        assert "one" in res
        assert "two" not in res
        assert "three" not in res
        assert "\n\n" not in res
        assert "more hidden" in res
        # double-check ncols
        assert all(len(i) == 50 for i in get_bar(res) if i.strip() and "more hidden" not in i)

    # all bars, leave=True
    with closing(StringIO()) as our_file:
        kwargs = {
            "file": our_file,
            "ncols": 50,
            "nrows": 2,
            "miniters": 0,
            "mininterval": 0,
        }
        with trange(10, desc="one", **kwargs) as t1:
            with trange(10, desc="two", **kwargs) as t2:
                assert "two" not in our_file.getvalue()
                with trange(10, desc="three", **kwargs) as t3:
                    assert "three" not in our_file.getvalue()
                    list(t3)
                list(t2)
            list(t1)

        res = our_file.getvalue()
        assert "one" in res
        assert "two" in res
        assert "three" in res
        assert "more hidden" in res
        # double-check ncols
        assert all(len(i) == 50 for i in get_bar(res) if i.strip() and "more hidden" not in i)

    # second bar becomes first, leave=False
    with closing(StringIO()) as our_file:
        kwargs = {
            "file": our_file,
            "ncols": 50,
            "nrows": 2,
            "miniters": 0,
            "mininterval": 0,
            "leave": False,
        }
        t1 = tldm(total=10, desc="one", **kwargs)
        with tldm(total=10, desc="two", **kwargs) as t2:
            t1.update()
            t2.update()
            t1.close()
            res = our_file.getvalue()
            assert "one" in res
            assert "two" not in res
            assert "more hidden" in res
            t2.update()

        res = our_file.getvalue()
        assert "two" in res


def test_initial() -> None:
    """Test `initial`"""
    with closing(StringIO()) as our_file:
        for _ in tldm(range(9), initial=10, total=19, file=our_file, miniters=1, mininterval=0):
            pass
        out = our_file.getvalue()
        assert "10/19" in out
        assert "19/19" in out


def test_colour() -> None:
    """Test `colour`"""
    with closing(StringIO()) as our_file:
        for _ in tldm(range(9), file=our_file, colour="#beefed"):
            pass
        out = our_file.getvalue()
        assert "\x1b[38;2;%d;%d;%dm" % (0xBE, 0xEF, 0xED) in out

        with catch_warnings(record=True) as w:
            simplefilter("always", category=TldmWarning)
            with tldm(total=1, file=our_file, colour="charm") as t:
                assert w
                t.update()
            assert "Unknown colour" in str(w[-1].message)

    with closing(StringIO()) as our_file2:
        for _ in tldm(range(9), file=our_file2, colour="blue"):
            pass
        out = our_file2.getvalue()
        assert "\x1b[34m" in out


def test_closed() -> None:
    """Test writing to closed file"""
    with closing(StringIO()) as our_file:
        for i in trange(9, file=our_file, miniters=1, mininterval=0):
            if i == 5:
                our_file.close()


def test_reversed(capsys):
    """Test reversed()"""
    expected_result = list(reversed(range(9)))
    real_result = list(reversed(tldm(range(9))))

    assert expected_result == real_result

    out, err = capsys.readouterr()
    assert not out
    assert "  0%" in err
    assert "100%" in err


def test_contains(capsys):
    """Test __contains__ doesn't iterate"""
    with tldm(list(range(9))) as t:
        assert 9 not in t
        assert all(i in t for i in range(9))
    out, err = capsys.readouterr()
    assert not out
    assert "  0%" in err
    assert "100%" not in err
