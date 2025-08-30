"""
Customisable progressbar decorator for iterators.
Includes a default `range` iterator printing to `stderr`.

Usage:
>>> from tqdm import trange, tqdm
>>> for i in trange(10):
...     ...
"""

import math
import numbers
from colorama import init  # TODO move to utils??
from icecream import ic

import signal
import sys
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
import copy
from datetime import datetime, timedelta, timezone
from numbers import Number
from operator import length_hint
from time import time
from warnings import warn
from typing import Any, Callable, Iterable, Iterator, Literal, TextIO, Self
from weakref import WeakSet

from ._monitor import TMonitor
from .utils import (
    CallbackIOWrapper,
    Comparable,
    DisableOnWriteError,
    FormatReplace,
    SimpleTextIOWrapper,
    _is_ascii,
    _screen_shape_wrapper,
    _supports_unicode,
    _term_move_up,
    disp_len,
    disp_trim,
    envwrap,
    get_ema_func,
    format_sizeof,
    format_meter,
    format_num,
    get_status_printer,
)


OSC_PROGRESS = "\x1b]9;4;1;"
OSC_END = "\7"

# Colorama initialization for windows
# TODO skip on linux
init()


class TqdmTypeError(TypeError):
    pass


class TqdmKeyError(KeyError):
    pass


class TqdmMonitorWarning:
    """tqdm monitor errors which do not affect external functionality"""

    pass


def TRLock(*args: tuple, **kwargs: dict[str, Any]) -> Any:
    """threading RLock"""
    try:
        from threading import RLock

        return RLock(*args, **kwargs)
    except (ImportError, OSError):  # pragma: no cover
        pass


class TqdmDefaultWriteLock(object):
    """
    Provide a default write lock for thread and multiprocessing safety.
    Works only on platforms supporting `fork` (so Windows is excluded).
    You must initialise a `tqdm` or `TqdmDefaultWriteLock` instance
    before forking in order for the write lock to work.
    On Windows, you need to supply the lock from the parent to the children as
    an argument to joblib or the parallelism lib you use.
    """

    # global thread lock so no setup required for multithreading.
    # NB: Do not create multiprocessing lock as it sets the multiprocessing
    # context, disallowing `spawn()`/`forkserver()`
    th_lock = TRLock()

    def __init__(self) -> None:
        # Create global parallelism locks to avoid racing issues with parallel
        # bars works only if fork available (Linux/MacOSX, but not Windows)
        cls = type(self)
        root_lock = cls.th_lock
        if root_lock is not None:
            root_lock.acquire()
        cls.create_mp_lock()
        self.locks = [lk for lk in [cls.mp_lock, cls.th_lock] if lk is not None]
        if root_lock is not None:
            root_lock.release()

    def acquire(self, *args: tuple, **kwargs: dict[str, Any]) -> None:
        for lock in self.locks:
            lock.acquire(*args, **kwargs)

    def release(self) -> None:
        # Release in inverse order of acquisition
        for lock in reversed(self.locks):
            lock.release()

    def __enter__(self) -> None:
        self.acquire()

    def __exit__(self, *args: tuple) -> None:
        self.release()

    def __del__(self) -> None:
        if TqdmDefaultWriteLock.mp_lock is not None:
            del TqdmDefaultWriteLock.mp_lock

    @classmethod
    def create_mp_lock(cls) -> None:
        if not hasattr(cls, "mp_lock"):
            try:
                from multiprocessing import RLock

                cls.mp_lock = RLock()
            except (ImportError, OSError):  # pragma: no cover
                cls.mp_lock = None


class tqdm(Comparable):
    """
    Decorate an iterable object, returning an iterator which acts exactly
    like the original iterable, but prints a dynamically updating
    progressbar every time a value is requested.

    Parameters
    ----------
    iterable  : iterable, optional
        Iterable to decorate with a progressbar.
        Leave blank to manually manage the updates.
    desc  : str, optional
        Prefix for the progressbar.
    total  : int or float, optional
        The number of expected iterations. If unspecified,
        len(iterable) is used if possible. If float("inf") or as a last
        resort, only basic progress statistics are displayed
        (no ETA, no progressbar).
        If `gui` is True and this parameter needs subsequent updating,
        specify an initial arbitrary large positive number,
        e.g. 9e9.
    leave  : bool, optional
        If [default: True], keeps all traces of the progressbar
        upon termination of iteration.
        If `None`, will leave only if `position` is `0`.
    file  : `io.TextIOWrapper` or `io.StringIO`, optional
        Specifies where to output the progress messages
        (default: sys.stderr). Uses `file.write(str)` and `file.flush()`
        methods.  For encoding, see `write_bytes`.
    ncols  : int, optional
        The width of the entire output message. If specified,
        dynamically resizes the progressbar to stay within this bound.
        If unspecified, attempts to use environment width. The
        fallback is a meter width of 10 and no limit for the counter and
        statistics. If 0, will not print any meter (only stats).
    mininterval  : float, optional
        Minimum progress display update interval [default: 0.1] seconds.
    maxinterval  : float, optional
        Maximum progress display update interval [default: 10] seconds.
        Automatically adjusts `miniters` to correspond to `mininterval`
        after long display update lag. Only works if `dynamic_miniters`
        or monitor thread is enabled.
    miniters  : int or float, optional
        Minimum progress display update interval, in iterations.
        If 0 and `dynamic_miniters`, will automatically adjust to equal
        `mininterval` (more CPU efficient, good for tight loops).
        If > 0, will skip display of specified number of iterations.
        Tweak this and `mininterval` to get very efficient loops.
        If your progress is erratic with both fast and slow iterations
        (network, skipping items, etc) you should set miniters=1.
    ascii  : bool or str, optional
        If unspecified or False, use unicode (smooth blocks) to fill
        the meter. The fallback is to use ASCII characters " 123456789#".
    disable  : bool, optional
        Whether to disable the entire progressbar wrapper
        [default: False]. If set to None, disable on non-TTY.
    unit  : str, optional
        String that will be used to define the unit of each iteration
        [default: it].
    unit_scale  : bool or int or float, optional
        If 1 or True, the number of iterations will be reduced/scaled
        automatically and a metric prefix following the
        International System of Units standard will be added
        (kilo, mega, etc.) [default: False]. If any other non-zero
        number, will scale `total` and `n`.
    dynamic_ncols  : bool, optional
        If set, constantly alters `ncols` and `nrows` to the
        environment (allowing for window resizes) [default: False].
    smoothing  : float, optional
        Exponential moving average smoothing factor for speed estimates
        (ignored in GUI mode). Ranges from 0 (average speed) to 1
        (current/instantaneous speed) [default: 0.3].
    bar_format  : str, optional
        Specify a custom bar string formatting. May impact performance.
        [default: '{l_bar}{bar}{r_bar}'], where
        l_bar='{desc}: {percentage:3.0f}%|' and
        r_bar='| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, '
            '{rate_fmt}{postfix}]'
        Possible vars: l_bar, bar, r_bar, n, n_fmt, total, total_fmt,
            percentage, elapsed, elapsed_s, ncols, nrows, desc, unit,
            rate, rate_fmt, rate_noinv, rate_noinv_fmt,
            rate_inv, rate_inv_fmt, postfix, unit_divisor,
            remaining, remaining_s, eta.
        Note that a trailing ": " is automatically removed after {desc}
        if the latter is empty.
    initial  : int or float, optional
        The initial counter value. Useful when restarting a progress
        bar [default: 0]. If using float, consider specifying `{n:.3f}`
        or similar in `bar_format`, or specifying `unit_scale`.
    position  : int, optional
        Specify the line offset to print this bar (starting from 0)
        Automatic if unspecified.
        Useful to manage multiple bars at once (eg, from threads).
    postfix  : dict or *, optional
        Specify additional stats to display at the end of the bar.
        Calls `set_postfix(**postfix)` if possible (dict).
    unit_divisor  : float, optional
        [default: 1000], ignored unless `unit_scale` is True.
    write_bytes  : bool, optional
        Whether to write bytes. If (default: False) will write unicode.
    lock_args  : tuple, optional
        Passed to `refresh` for intermediate output
        (initialisation, iterating, and updating).
    nrows  : int, optional
        The screen height. If specified, hides nested bars outside this
        bound. If unspecified, attempts to use environment height.
        The fallback is 20.
    colour  : str, optional
        Bar colour (e.g. 'green', '#00ff00').
    delay  : float, optional
        Don't display until [default: 0] seconds have elapsed.

    Returns
    -------
    out  : decorated iterator.
    """

    monitor_interval = 10  # set to 0 to disable the thread
    monitor = None
    _instances = WeakSet()

    registered_classes = set()

    def __new__(cls, *args: tuple, **kwargs: dict[str, Any]) -> Self:
        instance = object.__new__(cls)
        tqdm.registered_classes.add(cls)
        with cls.get_lock():  # also constructs lock if non-existent
            cls._instances.add(instance)
            # create monitoring thread
            if cls.monitor_interval and (
                cls.monitor is None or not cls.monitor.report()
            ):
                try:
                    cls.monitor = TMonitor(cls, cls.monitor_interval)
                except Exception as e:  # pragma: nocover
                    warn(
                        "tqdm:disabling monitor support"
                        " (monitor_interval = 0) due to:\n" + str(e),
                        TqdmMonitorWarning,
                        stacklevel=2,
                    )
                    cls.monitor_interval = 0
        return instance

    @classmethod
    def _get_free_pos(cls, instance=None):
        """Skips specified instance."""
        positions = {
            abs(inst.pos)
            for inst in cls._instances
            if inst is not instance and hasattr(inst, "pos")
        }
        return min(set(range(len(positions) + 1)).difference(positions))

    @classmethod
    def _decr_instances(cls, instance):
        """
        Remove from list and reposition another unfixed bar
        to fill the new gap.

        This means that by default (where all nested bars are unfixed),
        order is not maintained but screen flicker/blank space is minimised.
        (tqdm<=4.44.1 moved ALL subsequent unfixed bars up.)
        """
        with cls._lock:
            try:
                cls._instances.remove(instance)
            except KeyError:
                # if not instance.gui:  # pragma: no cover
                #     raise
                pass  # py2: maybe magically removed already
            # else:

            last = (instance.nrows or 20) - 1
            # find unfixed (`pos >= 0`) overflow (`pos >= nrows - 1`)
            instances = list(
                filter(lambda i: hasattr(i, "pos") and last <= i.pos, cls._instances)
            )
            # set first found to current `pos`
            if instances:
                inst = min(instances, key=lambda i: i.pos)
                inst.clear(nolock=True)
                inst.pos = abs(instance.pos)
            else:
                # renumber remaining bars with positions below this bar so
                # they maintain their positions
                apos = abs(instance.pos)
                readjust = [
                    (inst.pos, inst)
                    for inst in cls._instances
                    if not inst.disable and abs(getattr(inst, "pos", apos)) > apos
                ]
                for pos, inst in sorted(readjust, key=lambda pi: -abs(pi[0])):
                    newpos = inst.pos + (1 if pos < 0 else -1)
                    if newpos == 0 and inst.leave is None:
                        # any bars now moving to pos=0 should not be left on
                        # screen if `leave` was set to `None`.
                        inst.leave = False
                    if not inst.leave:
                        # Clear the old position before moving the bar so we
                        # don't leave any artefacts on screen.
                        inst.clear(nolock=True)
                    inst.pos = newpos
                    inst.display()

    @classmethod
    def write(cls, s, file=None, end="\n", nolock=False):
        """Print a message via tqdm (without overlap with bars)."""
        fp = file if file is not None else sys.stdout
        with cls.external_write_mode(file=file, nolock=nolock):
            # Write the message
            fp.write(s)
            fp.write(end)

    @classmethod
    def print(cls, *values, file=None, sep=" ", end="\n", nolock=False):
        """Print several heterogeneous values via tqdm (without overlap with bars)."""
        cls.write(
            sep.join("{}".format(v) for v in values), file=file, end=end, nolock=nolock
        )

    @classmethod
    @contextmanager
    def external_write_mode(cls, file=None, nolock=False):
        """
        Disable tqdm within context and refresh tqdm when exits.
        Useful when writing to standard output stream
        """
        fp = file if file is not None else sys.stdout

        try:
            if not nolock:
                cls.get_lock().acquire()
            # Clear all bars
            inst_cleared = []
            for inst in getattr(cls, "_instances", []):
                # Clear instance if in the target output file
                # or if write output + tqdm output are both either
                # sys.stdout or sys.stderr (because both are mixed in terminal)
                if hasattr(inst, "start_t") and (
                    inst.fp == fp
                    or all(f in (sys.stdout, sys.stderr) for f in (fp, inst.fp))
                ):
                    inst.clear(nolock=True)
                    inst_cleared.append(inst)
            yield
            # Force refresh display of bars we cleared
            for inst in inst_cleared:
                inst.refresh(nolock=True)
        finally:
            if not nolock:
                cls._lock.release()

    @classmethod
    def set_lock(cls, lock: TqdmDefaultWriteLock) -> None:
        """Set the global lock."""
        cls._lock = lock

    @classmethod
    def get_lock(cls) -> TqdmDefaultWriteLock:
        """Get the global lock. Construct it if it does not exist."""
        if not hasattr(cls, "_lock"):
            cls._lock = TqdmDefaultWriteLock()
        return cls._lock

    # override defaults via env vars
    @envwrap(
        "TQDM_",
        is_method=True,
        types={
            "total": float,
            "ncols": int,
            "miniters": float,
            "position": int,
            "nrows": int,
        },
    )
    def __init__(
        self,
        iterable: Iterable | None = None,
        desc: str | None = None,
        total: int | None = None,
        leave: bool = True,
        file: TextIO | None = None,
        ncols: int | None = None,
        mininterval: float = 0.1,
        maxinterval: float = 10.0,
        miniters: int | None = None,
        ascii: bool | str | None = None,
        disable: bool = False,
        unit: str = "it",
        unit_scale: bool = False,
        dynamic_ncols: bool = False,
        smoothing: float = 0.3,
        bar_format: str | None = None,
        initial: float | int = 0,
        position: int | None = None,
        postfix: str | None = None,
        unit_divisor: int = 1000,
        write_bytes: bool = False,
        lock_args: tuple | None = None,
        nrows: int | None = None,
        colour: str | None = None,
        delay: float = 0.0,
        title: str | None = False,
        **kwargs: dict[str, Any],
    ):
        """see tqdm.tqdm for arguments"""
        if file is None:
            file = sys.stderr

        if write_bytes:
            # Despite coercing unicode into bytes, py2 sys.std* streams
            # should have bytes written to them.
            file = SimpleTextIOWrapper(
                file, encoding=getattr(file, "encoding", None) or "utf-8"
            )

        file = DisableOnWriteError(file, tqdm_instance=self)

        if disable is None and hasattr(file, "isatty") and not file.isatty():
            disable = True

        if total is None and iterable is not None:
            # TODO do this via hasattr or some other better method than just trying a bunch
            # of random crap.
            try:
                total = len(iterable)
            except (TypeError, AttributeError):
                if hasattr(iterable, "__length_hint__"):
                    total = length_hint(iterable)
                else:
                    total = None
        if total == float("inf"):
            # Infinite iterations, behave same as unknown
            total = None

        if disable:
            self.iterable = iterable
            self.disable = disable
            with self.get_lock():
                self.pos = self._get_free_pos(self)
                self._instances.remove(self)
            self.n = initial
            self.total = total
            self.leave = leave
            return

        # Preprocess the arguments
        keep_original_size = ncols is not None, nrows is not None
        force_dynamic_ncols_update = dynamic_ncols
        if (
            (ncols is None or nrows is None) and (file in (sys.stderr, sys.stdout))
        ) or force_dynamic_ncols_update:  # pragma: no cover
            dynamic_ncols = _screen_shape_wrapper()
            if force_dynamic_ncols_update and dynamic_ncols:
                keep_original_size = False, False
                ncols, nrows = dynamic_ncols(file)
            else:
                if dynamic_ncols:
                    _ncols, _nrows = dynamic_ncols(file)
                    if ncols is None:
                        ncols = _ncols
                    if nrows is None:
                        nrows = _nrows

        if miniters is None:
            miniters = 0
            dynamic_miniters = True
        else:
            dynamic_miniters = False

        if mininterval is None:
            mininterval = 0

        if maxinterval is None:
            maxinterval = 0

        if ascii is None:
            ascii = not _supports_unicode(file)

        if bar_format and ascii is not True and not _is_ascii(ascii):
            # Convert bar format into unicode since terminal uses unicode
            bar_format = str(bar_format)

        if smoothing is None:
            smoothing = 0

        # Store the arguments
        self.iterable = iterable
        self.desc = desc or ""
        self.total = total
        self.leave = leave
        self.fp = file
        self.ncols = ncols
        self.nrows = nrows
        self.keep_original_size = keep_original_size
        self.mininterval = mininterval
        self.maxinterval = maxinterval
        self.miniters = miniters
        self.dynamic_miniters = dynamic_miniters
        self.ascii = ascii
        self.disable = disable
        self.unit = unit
        self.unit_scale = unit_scale
        self.unit_divisor = unit_divisor
        self.initial = initial
        self.lock_args = lock_args
        self.delay = delay
        self.force_dynamic_ncols_update = force_dynamic_ncols_update
        self.dynamic_ncols = dynamic_ncols
        self.smoothing = smoothing
        self._ema_dn = get_ema_func(smoothing)
        self._ema_dt = get_ema_func(smoothing)
        self._ema_miniters = get_ema_func(smoothing)
        self.bar_format = bar_format
        self.postfix = None
        self.colour = colour
        self._time = time
        self.title = title
        if postfix:
            try:
                self.set_postfix(refresh=False, **postfix)
            except TypeError:
                self.postfix = postfix

        # Init the iterations counters
        self.last_print_n = initial
        self.n = initial

        # if nested, at initial sp() call we replace '\r' by '\n' to
        # not overwrite the outer progress bar
        with self.get_lock():
            # mark fixed positions as negative
            self.pos = self._get_free_pos(self) if position is None else -position

        # Initialize the screen printer
        self.sp = get_status_printer(self.fp)
        if delay <= 0:
            self.refresh(lock_args=self.lock_args)

        # Init the time counter
        self.last_print_t = self._time()
        self.last_pause_t = 0.0
        # NB: Avoid race conditions by setting start_t at the very end of init
        self.start_t = self.last_print_t

    def __bool__(self) -> bool:
        if self.total is not None:
            return self.total > 0
        if self.iterable is None:
            raise TypeError("bool() undefined when iterable == total == None")
        return bool(self.iterable)

    def __len__(self) -> int:
        return (
            self.total
            if self.iterable is None
            else self.iterable.shape[0]
            if hasattr(self.iterable, "shape")
            else len(self.iterable)
            if hasattr(self.iterable, "__len__")
            else length_hint(self.iterable)
            if hasattr(self.iterable, "__length_hint__")
            else getattr(self, "total", None)
        )

    def __reversed__(self) -> Self:
        try:
            orig = self.iterable
        except AttributeError:
            raise TypeError("'tqdm' object is not reversible")
        else:
            # Shallow copies the object.
            reversed_obj = copy.copy(self)

            # Replaces the iterable with the reversed iterable.
            reversed_obj.iterable = reversed(self.iterable)

            return reversed_obj
        finally:
            self.iterable = orig

    def __contains__(self, item: Any) -> bool:
        contains = getattr(self.iterable, "__contains__", None)
        return contains(item) if contains is not None else item in self.__iter__()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            self.close()
        except AttributeError:
            # maybe eager thread cleanup upon external error
            if (exc_type, exc_value, traceback) == (None, None, None):
                raise
            warn("AttributeError ignored", TqdmWarning, stacklevel=2)

    # TODO add the functionality from this code back if necessary
    # def __del__(self) -> None:
    #     self.close()
    #     if len(tqdm._instances) == 0:
    #         if hasattr(tqdm, "_lock"):
    #             del tqdm._lock
    #         if hasattr(tqdm, "monitor") and tqdm.monitor is not None:
    #             tqdm.monitor.exit()

    def __str__(self) -> str:
        return format_meter(**self.format_dict)

    @property
    def _comparable(self) -> int:
        return abs(getattr(self, "pos", 1 << 31))

    def __hash__(self) -> int:
        return id(self)

    def __iter__(self) -> Iterator:
        """Backward-compatibility to use: for x in tqdm(iterable)"""

        # Inlining instance variables as locals (speed optimisation)
        iterable = self.iterable

        # If the bar is disabled, then just walk the iterable
        # (note: keep this check outside the loop for performance)
        if self.disable:
            for obj in iterable:
                yield obj
            return

        # NOTE these assignments are actually necessary because instance variables get messed with
        mininterval = self.mininterval
        last_print_t = self.last_print_t
        last_print_n = self.last_print_n
        min_start_t = self.start_t + self.delay
        n = self.n
        time = self._time

        try:
            for obj in iterable:
                yield obj
                # Update and possibly print the progressbar.
                # Note: does not call self.update(1) for speed optimisation.
                n += 1

                if n - last_print_n >= self.miniters:
                    cur_t = time()
                    dt = cur_t - last_print_t
                    if dt >= mininterval and cur_t >= min_start_t:
                        self.update(n - last_print_n)
                        last_print_n = self.last_print_n
                        last_print_t = self.last_print_t
        finally:
            self.n = n
            self.close()

    def update(self, n: int | float = 1) -> None:
        """
        # TODO fix the type annotation here
        Manually update the progress bar, useful for streams
        such as reading files.
        E.g.:
        >>> t = tqdm(total=filesize) # Initialise
        >>> for current_buffer in stream:
        ...    ...
        ...    t.update(len(current_buffer))
        >>> t.close()
        The last line is highly recommended, but possibly not necessary if
        `t.update()` will be called in such a way that `filesize` will be
        exactly reached and printed.

        Parameters
        ----------
        n  : int or float, optional
            Increment to add to the internal counter of iterations
            [default: 1]. If using float, consider specifying `{n:.3f}`
            or similar in `bar_format`, or specifying `unit_scale`.

        Returns
        -------
        out  : bool or None
            True if a `display()` was triggered.
        """
        if self.disable:
            return

        if n < 0:
            self.last_print_n += n  # for auto-refresh logic to work
        self.n += n

        # check counter first to reduce calls to time()
        if self.n - self.last_print_n >= self.miniters:
            cur_t = self._time()
            dt = cur_t - self.last_print_t
            if dt >= self.mininterval and cur_t >= self.start_t + self.delay:
                cur_t = self._time()
                dn = self.n - self.last_print_n  # >= n
                if self.smoothing and dt and dn:
                    # EMA (not just overall average)
                    self._ema_dn(dn)
                    self._ema_dt(dt)
                self.refresh(lock_args=self.lock_args)
                if self.dynamic_miniters:
                    # If no `miniters` was specified, adjust automatically to the
                    # maximum iteration rate seen so far between two prints.
                    # e.g.: After running `tqdm.update(5)`, subsequent
                    # calls to `tqdm.update()` will only cause an update after
                    # at least 5 more iterations.
                    if self.maxinterval and dt >= self.maxinterval:
                        self.miniters = dn * (self.mininterval or self.maxinterval) / dt
                    elif self.smoothing:
                        # EMA miniters update
                        self.miniters = self._ema_miniters(
                            dn
                            * (self.mininterval / dt if self.mininterval and dt else 1)
                        )
                    else:
                        # max iters between two prints
                        self.miniters = max(self.miniters, dn)

                # Store old values for next call
                self.last_print_n = self.n
                self.last_print_t = cur_t
                return True

    def close(self) -> None:
        """Cleanup and (if leave=False) close the progressbar."""
        if self.disable:
            return

        # Prevent multiple closures
        self.disable = True

        try:
            if self.last_print_t < self.start_t + self.delay:
                # haven't ever displayed; nothing to clear
                return

            # annoyingly, _supports_unicode isn't good enough
            def fp_write(s):
                self.fp.write(str(s))

            try:
                fp_write("")
            except ValueError as e:
                if "closed" in str(e):
                    return
                raise  # pragma: no cover

            pos = abs(self.pos)
            leave = pos == 0 if self.leave is None else self.leave

            with self._lock:
                if leave:
                    # stats for overall rate (no weighted average)
                    self._ema_dt = lambda: None
                    self.display(pos=0)
                    fp_write("\n")
                else:
                    # clear previous display
                    if self.display(msg="", pos=pos) and not pos:
                        fp_write("\r")

        finally:
            # decrement instance pos and remove from internal set
            self._decr_instances(self)

    def clear(self, nolock: bool = False) -> None:
        """Clear current bar display."""
        if self.disable:
            return

        if not nolock:
            self._lock.acquire()
        pos = abs(self.pos)
        try:
            if pos < (self.nrows or 20):
                self._moveto(pos)
                self.sp("")
                self.fp.write("\r")  # place cursor back at the beginning of line
                self._moveto(-pos)
        finally:
            if not nolock:
                self._lock.release()

    def refresh(self, nolock: bool = False, lock_args: tuple = None) -> None:
        """
        Force refresh the display of this bar.

        Parameters
        ----------
        nolock  : bool, optional
            If `True`, does not lock.
            If [default: `False`]: calls `acquire()` on internal lock.
        lock_args  : tuple, optional
            Passed to internal lock's `acquire()`.
            If specified, will only `display()` if `acquire()` returns `True`.
        """
        if self.disable:
            return

        if not nolock:
            if lock_args:
                if not self._lock.acquire(*lock_args):
                    return False
            else:
                self._lock.acquire()
        try:
            self.display()
        finally:
            if not nolock:
                self._lock.release()

    def pause(self, refresh: bool = True) -> None:
        """Pause the tqdm timer.

        Refresh the progress bar by default.
        """
        if self.disable:
            return

        if self.last_pause_t != 0.0:
            warn("The progress bar is already paused")
            return

        if refresh:  # By default refresh before doing a pause
            self.refresh()

        self.last_pause_t = self._time()

    def unpause(self) -> None:
        """Restart tqdm timer from last pause."""
        if self.disable:
            return

        if self.last_pause_t == 0.0:
            warn("The progress bar is not paused")
            return

        dt = self._time() - self.last_pause_t
        self.last_pause_t = 0.0

        if dt < 0.0:
            return

        self.start_t += dt
        self.last_print_t += dt

    def reset(self, total: int | float | None = None) -> None:
        """
        Resets to 0 iterations for repeated use.

        Consider combining with `leave=True`.

        Parameters
        ----------
        total  : int or float, optional. Total to use for the new bar.
        """
        self.n = 0
        if total is not None:
            self.total = total
        if self.disable:
            return
        self.last_print_n = 0
        self.last_print_t = self.start_t = self._time()
        self._ema_dn = get_ema_func(self.smoothing)
        self._ema_dt = get_ema_func(self.smoothing)
        self._ema_miniters = get_ema_func(self.smoothing)
        self.refresh()

    def set_description(self, desc: str | None = None, refresh: bool = True) -> None:
        """
        Set/modify description of the progress bar.

        Parameters
        ----------
        desc  : str, optional
        refresh  : bool, optional
            Forces refresh [default: True].
        """
        self.desc = desc + ": " if desc else ""
        if refresh:
            self.refresh()

    def set_description_str(
        self, desc: str | None = None, refresh: bool = True
    ) -> None:
        """Set/modify description without ': ' appended."""
        self.desc = desc or ""
        if refresh:
            self.refresh()

    def set_postfix(
        self, ordered_dict: dict = None, refresh: bool = True, **kwargs: dict[str, Any]
    ) -> None:
        """
        Set/modify postfix (additional stats)
        with automatic formatting based on datatype.

        Parameters
        ----------
        ordered_dict  : dict, optional
        refresh  : bool, optional
            Forces refresh [default: True].
        kwargs  : dict, optional
        """
        # Sort in alphabetical order to be more deterministic
        postfix = OrderedDict([] if ordered_dict is None else ordered_dict)
        for key in sorted(kwargs.keys()):
            postfix[key] = kwargs[key]
        # Preprocess stats according to datatype
        for key in postfix.keys():
            # Number: limit the length of the string
            if isinstance(postfix[key], Number):
                postfix[key] = format_num(postfix[key])
            # Else for any other type, try to get the string conversion
            elif not isinstance(postfix[key], str):
                postfix[key] = str(postfix[key])
            # Else if it's a string, don't need to preprocess anything
        # Stitch together to get the final postfix
        self.postfix = ", ".join(
            key + "=" + postfix[key].strip() for key in postfix.keys()
        )
        if refresh:
            self.refresh()

    def set_postfix_str(self, s: str = "", refresh: bool = True) -> None:
        """
        Postfix without dictionary expansion, similar to prefix handling.
        """
        self.postfix = str(s)
        if refresh:
            self.refresh()

    def _moveto(self, n: int) -> None:
        # TODO: private method
        self.fp.write("\n" * n + _term_move_up() * -n)
        getattr(self.fp, "flush", lambda: None)()

    @property
    def format_dict(self) -> dict[str, Any]:
        """Public API for read-only member access."""
        if self.disable and not hasattr(self, "unit"):
            return defaultdict(
                lambda: None,
                {"n": self.n, "total": self.total, "elapsed": 0, "unit": "it"},
            )
        if self.force_dynamic_ncols_update and self.dynamic_ncols:
            self.ncols, self.nrows = self.dynamic_ncols(self.fp)
        return {
            "n": self.n,
            "total": self.total,
            "elapsed": self._time() - self.start_t if hasattr(self, "start_t") else 0,
            "ncols": self.ncols,
            "nrows": self.nrows,
            "prefix": self.desc,
            "ascii": self.ascii,
            "unit": self.unit,
            "unit_scale": self.unit_scale,
            "rate": self._ema_dn() / self._ema_dt() if self._ema_dt() else None,
            "bar_format": self.bar_format,
            "postfix": self.postfix,
            "unit_divisor": self.unit_divisor,
            "initial": self.initial,
            "colour": self.colour,
            "title": self.title,
        }

    def display(self, msg: str | None = None, pos: int | None = None) -> None:
        """
        Use `self.sp` to display `msg` in the specified `pos`.

        Consider overloading this function when inheriting to use e.g.:
        `self.some_frontend(**self.format_dict)` instead of `self.sp`.

        Parameters
        ----------
        msg  : str, optional. What to display (default: `repr(self)`).
        pos  : int, optional. Position to `moveto`
          (default: `abs(self.pos)`).
        """
        if pos is None:
            pos = abs(self.pos)

        nrows = self.nrows or 20
        if pos >= nrows - 1:
            if pos >= nrows:
                return False
            if msg or msg is None:  # override at `nrows - 1`
                msg = " ... (more hidden) ..."

        if pos:
            self._moveto(pos)
        self.sp(self.__str__() if msg is None else msg)
        if pos:
            self._moveto(-pos)
        return True

    @classmethod
    @contextmanager
    def wrapattr(
        cls,
        stream: TextIO,
        method: Literal["read", "write"],
        total: int | float | None = None,
        bytes: bool = True,
        **tqdm_kwargs: dict[str, Any],
    ):
        # TODO add return type
        """
        stream  : file-like object.
        method  : str, "read" or "write". The result of `read()` and
            the first argument of `write()` should have a `len()`.

        >>> with tqdm.wrapattr(file_obj, "read", total=file_obj.size) as fobj:
        ...     while True:
        ...         chunk = fobj.read(chunk_size)
        ...         if not chunk:
        ...             break
        """
        with cls(total=total, **tqdm_kwargs) as t:
            if bytes:
                t.unit = "B"
                t.unit_scale = True
                t.unit_divisor = 1024
            yield CallbackIOWrapper(t.update, stream, method)

    @classmethod
    def pandas(cls, **tqdm_kwargs: dict[str, Any]) -> None:
        """
        Registers the current `tqdm` class with
            pandas.core.
            ( frame.DataFrame
            | series.Series
            | groupby.(generic.)DataFrameGroupBy
            | groupby.(generic.)SeriesGroupBy
            ).progress_apply

        A new instance will be created every time `progress_apply` is called,
        and each instance will automatically `close()` upon completion.

        Parameters
        ----------
        tqdm_kwargs  : arguments for the tqdm instance

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from tqdm import tqdm
        >>> from tqdm.gui import tqdm as tqdm_gui
        >>>
        >>> df = pd.DataFrame(np.random.randint(0, 100, (100000, 6)))
        >>> tqdm.pandas(ncols=50)  # can use tqdm_gui, optional kwargs, etc
        >>> # Now you can use `progress_apply` instead of `apply`
        >>> df.groupby(0).progress_apply(lambda x: x**2)

        References
        ----------
        <https://stackoverflow.com/questions/18603270/\
        progress-indicator-during-pandas-operations-python>
        """
        from warnings import catch_warnings, simplefilter

        from pandas.core.frame import DataFrame
        from pandas.core.series import Series

        try:
            with catch_warnings():
                simplefilter("ignore", category=FutureWarning)
                from pandas import Panel
        except ImportError:  # pandas>=1.2.0
            Panel = None
        Rolling, Expanding = None, None
        try:  # pandas>=1.0.0
            from pandas.core.window.rolling import _Rolling_and_Expanding
        except ImportError:
            try:  # pandas>=0.18.0
                from pandas.core.window import _Rolling_and_Expanding
            except ImportError:  # pandas>=1.2.0
                try:  # pandas>=1.2.0
                    from pandas.core.window.expanding import Expanding
                    from pandas.core.window.rolling import Rolling

                    _Rolling_and_Expanding = Rolling, Expanding
                except ImportError:  # pragma: no cover
                    _Rolling_and_Expanding = None
        try:  # pandas>=0.25.0
            from pandas.core.groupby.generic import SeriesGroupBy  # , NDFrameGroupBy
            from pandas.core.groupby.generic import DataFrameGroupBy
        except ImportError:  # pragma: no cover
            try:  # pandas>=0.23.0
                from pandas.core.groupby.groupby import DataFrameGroupBy, SeriesGroupBy
            except ImportError:
                from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy
        try:  # pandas>=0.23.0
            from pandas.core.groupby.groupby import GroupBy
        except ImportError:  # pragma: no cover
            from pandas.core.groupby import GroupBy

        try:  # pandas>=0.23.0
            from pandas.core.groupby.groupby import PanelGroupBy
        except ImportError:
            try:
                from pandas.core.groupby import PanelGroupBy
            except ImportError:  # pandas>=0.25.0
                PanelGroupBy = None

        tqdm_kwargs = tqdm_kwargs.copy()
        deprecated_t = [tqdm_kwargs.pop("deprecated_t", None)]

        def inner_generator(df_function="apply"):
            def inner(df, func, *args, **kwargs):
                """
                Parameters
                ----------
                df  : (DataFrame|Series)[GroupBy]
                    Data (may be grouped).
                func  : function
                    To be applied on the (grouped) data.
                **kwargs  : optional
                    Transmitted to `df.apply()`.
                """

                # Precompute total iterations
                total = tqdm_kwargs.pop("total", getattr(df, "ngroups", None))
                if total is None:  # not grouped
                    if df_function == "applymap":
                        total = df.size
                    elif isinstance(df, Series):
                        total = len(df)
                    elif _Rolling_and_Expanding is None or not isinstance(
                        df, _Rolling_and_Expanding
                    ):
                        # DataFrame or Panel
                        axis = kwargs.get("axis", 0)
                        if axis == "index":
                            axis = 0
                        elif axis == "columns":
                            axis = 1
                        # when axis=0, total is shape[axis1]
                        total = df.size // df.shape[axis]

                # Init bar
                if deprecated_t[0] is not None:
                    t = deprecated_t[0]
                    deprecated_t[0] = None
                else:
                    t = cls(total=total, **tqdm_kwargs)

                if len(args) > 0:
                    # *args intentionally not supported (see #244, #299)
                    TqdmDeprecationWarning(
                        "Except func, normal arguments are intentionally"
                        + " not supported by"
                        + " `(DataFrame|Series|GroupBy).progress_apply`."
                        + " Use keyword arguments instead.",
                        fp_write=getattr(t.fp, "write", sys.stderr.write),
                    )

                try:  # pandas>=1.3.0
                    from pandas.core.common import is_builtin_func
                except ImportError:
                    is_builtin_func = df._is_builtin_func
                try:
                    func = is_builtin_func(func)
                except TypeError:
                    pass

                # Define bar updating wrapper
                def wrapper(*args, **kwargs):
                    # update tbar correctly
                    # it seems `pandas apply` calls `func` twice
                    # on the first column/row to decide whether it can
                    # take a fast or slow code path; so stop when t.total==t.n
                    t.update(n=1 if not t.total or t.n < t.total else 0)
                    return func(*args, **kwargs)

                # Apply the provided function (in **kwargs)
                # on the df using our wrapper (which provides bar updating)
                try:
                    return getattr(df, df_function)(wrapper, **kwargs)
                finally:
                    t.close()

            return inner

        # Monkeypatch pandas to provide easy methods
        # Enable custom tqdm progress in pandas!
        Series.progress_apply = inner_generator()
        SeriesGroupBy.progress_apply = inner_generator()
        Series.progress_map = inner_generator("map")
        SeriesGroupBy.progress_map = inner_generator("map")

        DataFrame.progress_apply = inner_generator()
        DataFrameGroupBy.progress_apply = inner_generator()
        DataFrame.progress_applymap = inner_generator("applymap")
        DataFrame.progress_map = inner_generator("map")
        DataFrameGroupBy.progress_map = inner_generator("map")

        if Panel is not None:
            Panel.progress_apply = inner_generator()
        if PanelGroupBy is not None:
            PanelGroupBy.progress_apply = inner_generator()

        GroupBy.progress_apply = inner_generator()
        GroupBy.progress_aggregate = inner_generator("aggregate")
        GroupBy.progress_transform = inner_generator("transform")

        if Rolling is not None and Expanding is not None:
            Rolling.progress_apply = inner_generator()
            Expanding.progress_apply = inner_generator()
        elif _Rolling_and_Expanding is not None:
            _Rolling_and_Expanding.progress_apply = inner_generator()


def trange(*args: tuple, **kwargs: dict[str, Any]) -> tqdm:
    """Shortcut for tqdm(range(*args), **kwargs)."""
    return tqdm(range(*args), **kwargs)


def resize_signal_handler(signalnum, frame):
    for cls in tqdm.registered_classes:
        with cls.get_lock():
            for instance in cls._instances:
                if instance.dynamic_ncols:
                    ncols, nrows = instance.dynamic_ncols(instance.fp)
                    if not instance.keep_original_size[0]:
                        instance.ncols = ncols
                    if not instance.keep_original_size[1]:
                        instance.nrows = nrows


try:
    signal.signal(signal.SIGWINCH, resize_signal_handler)
except AttributeError:
    pass  # Some systems, like Windows, do not have SIGWINCH
