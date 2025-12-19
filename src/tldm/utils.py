"""
General helpers required for `tldm.std`.
"""

import contextlib
import math
import numbers
import os
import re
import sys
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from functools import wraps
from typing import Any, TextIO, TypeVar, cast
from warnings import warn
from weakref import proxy

from colorama import Fore, Style, init, just_fix_windows_console
from wcwidth import wcwidth

CUR_OS = sys.platform
IS_WIN = CUR_OS.startswith(("win32", "cygwin"))
IS_NIX = CUR_OS.startswith(("aix", "linux", "darwin", "freebsd"))
RE_ANSI = re.compile(r"\x1b\[[;\d]*[A-Za-z]")
T = TypeVar("T")

if IS_WIN:
    init()
    just_fix_windows_console()


class FormatReplace:
    """
    >>> a = FormatReplace('something')
    >>> f"{a:5d}"
    'something'
    """

    def __init__(self, replace: str = "") -> None:
        self.replace = replace
        self.format_called = 0

    def __format__(self, _: str) -> str:
        self.format_called += 1
        return self.replace


class ObjectWrapper:
    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    def __setattr__(self, name: str, value: Any) -> None:
        return setattr(self._wrapped, name, value)

    def wrapper_getattr(self, name: str) -> Any:
        """Actual `self.getattr` rather than self._wrapped.getattr"""
        return object.__getattr__(self, name)  # type: ignore[attr-defined]

    def wrapper_setattr(self, name: str, value: Any) -> None:
        """Actual `self.setattr` rather than self._wrapped.setattr"""
        return object.__setattr__(self, name, value)

    def __init__(self, wrapped: Any) -> None:
        """
        Thin wrapper around a given object
        """
        self.wrapper_setattr("_wrapped", wrapped)


class DisableOnWriteError(ObjectWrapper):
    """
    Disable the given `tldm_instance` upon `write()` or `flush()` errors.
    """

    @staticmethod
    def disable_on_exception(tldm_instance: Any, func: Callable[..., Any]) -> Callable[..., Any]:
        """
        Quietly set `tldm_instance.miniters=inf` if `func` raises `errno=5`.
        """
        tldm_instance = proxy(tldm_instance)

        def inner(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except OSError as e:
                if e.errno != 5:
                    raise
                with contextlib.suppress(ReferenceError):
                    tldm_instance.miniters = float("inf")
            except ValueError as e:
                if "closed" not in str(e):
                    raise
                with contextlib.suppress(ReferenceError):
                    tldm_instance.miniters = float("inf")

        return inner

    def __init__(self, wrapped: TextIO, tldm_instance: Any) -> None:
        super().__init__(wrapped)
        if hasattr(wrapped, "write"):
            self.wrapper_setattr("write", self.disable_on_exception(tldm_instance, wrapped.write))
        if hasattr(wrapped, "flush"):
            self.wrapper_setattr("flush", self.disable_on_exception(tldm_instance, wrapped.flush))

    def __eq__(self, other: Any) -> bool:
        return bool(self._wrapped == getattr(other, "_wrapped", other))


class CallbackIOWrapper(ObjectWrapper):
    def __init__(
        self, callback: Callable[[int], None], stream: TextIO, method: str = "read"
    ) -> None:
        """
        Wrap a given `file`-like object's `read()` or `write()` to report
        lengths to the given `callback`
        """
        super().__init__(stream)
        func = getattr(stream, method)
        if method == "write":

            @wraps(func)
            def write(data, *args, **kwargs):
                res = func(data, *args, **kwargs)
                callback(len(data))
                return res

            self.wrapper_setattr("write", write)
        elif method == "read":

            @wraps(func)
            def read(*args, **kwargs):
                data = func(*args, **kwargs)
                callback(len(data))
                return data

            self.wrapper_setattr("read", read)
        else:
            raise KeyError("Can only wrap read/write methods")


def _is_utf(encoding: str) -> bool:
    try:
        "\u2588\u2589".encode(encoding)
    except UnicodeEncodeError:
        return False
    except Exception:
        try:
            return encoding.lower().startswith("utf-") or (encoding == "U8")
        except Exception:
            return False
    else:
        return True


def _supports_unicode(fp: Any) -> bool:
    try:
        return _is_utf(fp.encoding)
    except AttributeError:
        return False


def _is_ascii(s: Any) -> bool:
    if isinstance(s, str):
        return all(ord(c) <= 255 for c in s)
    return _supports_unicode(s)


def _screen_shape_wrapper() -> Callable[[TextIO], tuple[int, int] | tuple[None, None]] | None:
    """
    Return a function which returns console dimensions (width, height).
    Supported: linux, osx, windows, cygwin.
    """
    _screen_shape = None
    if IS_WIN:
        _screen_shape = _screen_shape_windows
        if _screen_shape is None:
            _screen_shape = _screen_shape_tput
    if IS_NIX:
        _screen_shape = _screen_shape_linux
    return _screen_shape


def _screen_shape_windows(
    fp: TextIO,
) -> tuple[int, int] | tuple[None, None]:  # pragma: no cover
    try:
        import struct
        from ctypes import (  # type: ignore[attr-defined,misc,unused-ignore]
            create_string_buffer,
            windll,
        )
        from sys import stdin, stdout

        io_handle = -12  # assume stderr
        if fp == stdin:
            io_handle = -10
        elif fp == stdout:
            io_handle = -11

        h = windll.kernel32.GetStdHandle(io_handle)
        csbi = create_string_buffer(22)
        res = windll.kernel32.GetConsoleScreenBufferInfo(h, csbi)
        if res:
            (
                _bufx,
                _bufy,
                _curx,
                _cury,
                _wattr,
                left,
                top,
                right,
                bottom,
                _maxx,
                _maxy,
            ) = struct.unpack("hhhhHhhhhhh", csbi.raw)
            return right - left, bottom - top  # +1
    except Exception:  # nosec
        pass
    return None, None


def _screen_shape_tput(
    *_: Any,
) -> tuple[int, int] | tuple[None, None]:  # pragma: no cover
    """cygwin xterm (windows)"""
    try:
        import shlex
        from subprocess import check_call  # nosec

        return cast(
            tuple[int, int],
            tuple(int(check_call(shlex.split("tput " + i))) - 1 for i in ("cols", "lines")),
        )
    except Exception:  # nosec
        pass
    return None, None


def _screen_shape_linux(
    fp: TextIO,
) -> tuple[int, int] | tuple[None, None]:
    try:
        from array import array
        from fcntl import ioctl  # type: ignore[import,attr-defined,unused-ignore]
        from termios import TIOCGWINSZ  # type: ignore[import,attr-defined,unused-ignore]
    except ImportError:
        return None, None
    else:
        try:
            rows, cols = array("h", ioctl(fp, TIOCGWINSZ, ("\0" * 8).encode()))[:2]
            return cols, rows
        except Exception:
            try:
                return cast(
                    tuple[int, int],
                    tuple(int(os.environ[i]) - 1 for i in ("COLUMNS", "LINES")),
                )
            except (KeyError, ValueError):
                return None, None


def _wcswidth_tolerant(pwcs: str, n: int | None = None, unicode_version: str = "auto") -> int:
    """
    Given a unicode string, return its printable length on a terminal.

    Modified fault-tolerant version of ``wcwidth.wcswidth`` function.
    Non-printable characters are considered zero-length.

    :param str pwcs: Measure width of given unicode string.
    :param int n: When ``n`` is None (default), return the length of the
        entire string, otherwise width the first ``n`` characters specified.
    :param str unicode_version: An explicit definition of the unicode version
        level to use for determination, may be ``auto`` (default), which uses
        the Environment Variable, ``UNICODE_VERSION`` if defined, or the latest
        available unicode version, otherwise.
    :rtype: int
    :returns: The width, in cells, necessary to display the first ``n``
        characters of the unicode string ``pwcs``.
    """
    # pylint: disable=C0103
    #         Invalid argument name "n"

    end = len(pwcs) if n is None else n
    idx = slice(0, end)
    width = 0
    for char in pwcs[idx]:
        wcw = wcwidth(char, unicode_version)
        if wcw >= 0:
            width += wcw
    return width


def disp_len(data: str) -> int:
    """
    Returns the real on-screen length of a string which may contain
    ANSI control codes and wide chars.
    """
    return _wcswidth_tolerant(RE_ANSI.sub("", data))


def disp_trim(data: str, length: int) -> str:
    """
    Trim a string which may contain ANSI control characters.
    """
    if len(data) == disp_len(data):
        return data[:length]

    ansi_present = bool(RE_ANSI.search(data))
    while disp_len(data) > length:  # carefully delete one char at a time
        data = data[:-1]
    if ansi_present and bool(RE_ANSI.search(data)):
        # assume ANSI reset is required
        return data if data.endswith("\033[0m") else data + "\033[0m"
    return data


def get_ema_func(smoothing: float = 0.3) -> Callable[[float | None], float]:
    """
    Get a function that computes the exponential moving average (EMA) of a stream of values.

    Parameters
    ----------
    smoothing  : float, optional
        Smoothing factor in range [0, 1], [default: 0.3].
        Increase to give more weight to recent values.
        Ranges from 0 (yields old value) to 1 (yields new value).

    Returns
    -------
    Callable[[float | None], float]
        A function that takes a new value and returns the current EMA.
    """
    beta = 1 - smoothing
    last: float = 0
    calls: int = 0

    def ema(x: float | None = None) -> float:
        nonlocal last, calls
        if x is not None:
            last = smoothing * x + beta * last
            calls += 1
        return last / (1 - beta**calls) if calls else last

    return ema


###### TODO all of the helper functions below are related to formatting. #####
# should probably be moved to a separate file
# TODO make a separate file with exception + warning types


class TldmWarning(Warning):
    """base class for all tldm warnings.

    Used for non-external-code-breaking errors, such as garbled printing.
    """

    def __init__(
        self,
        msg: str,
        fp_write: Callable[[str], None] | None = None,
        *args: Any,
        **kwargs: dict[str, Any],
    ) -> None:
        if fp_write is not None:
            fp_write("\n" + self.__class__.__name__ + ": " + str(msg).rstrip() + "\n")
        else:
            super().__init__(msg, *args, **kwargs)


class Bar:
    """
    `str.format`-able bar with format specifiers: `[width][type]`

    - `width`
      + unspecified (default): use `self.default_len`
      + `int >= 0`: overrides `self.default_len`
      + `int < 0`: subtract from `self.default_len`
    - `type`
      + `a`: ascii (`charset=self.ASCII` override)
      + `u`: unicode (`charset=self.UTF` override)
      + `b`: blank (`charset="  "` override)
    """

    ASCII = " 123456789#"
    UTF = " " + "".join(map(chr, range(0x258F, 0x2587, -1)))
    BLANK = "  "
    COLOUR_RESET = Style.RESET_ALL
    COLOUR_RGB = "\x1b[38;2;%d;%d;%dm"
    COLOURS = {
        "BLACK": Fore.BLACK,
        "RED": Fore.RED,
        "GREEN": Fore.GREEN,
        "YELLOW": Fore.YELLOW,
        "BLUE": Fore.BLUE,
        "MAGENTA": Fore.MAGENTA,
        "CYAN": Fore.CYAN,
        "WHITE": Fore.WHITE,
    }

    def __init__(
        self,
        frac: float,
        default_len: int = 10,
        charset: str = UTF,
        colour: str | None = None,
    ) -> None:
        if not 0 <= frac <= 1:
            warn("clamping frac to range [0, 1]", TldmWarning, stacklevel=2)
            frac = max(0, min(1, frac))
        assert default_len > 0
        self.frac = frac
        self.default_len = default_len
        self.charset = charset
        self.colour = colour

    @property
    def colour(self) -> str | None:
        return self._colour

    @colour.setter
    def colour(self, value: str | None) -> None:
        """
        TODO change to use color setting and validation with colorama
        """
        if not value:
            self._colour = None
            return
        try:
            if value.upper() in self.COLOURS:
                self._colour = self.COLOURS[value.upper()]
            elif value[0] == "#" and len(value) == 7:
                self._colour = self.COLOUR_RGB % tuple(
                    int(i, 16) for i in (value[1:3], value[3:5], value[5:7])
                )
            else:
                raise KeyError
        except (KeyError, AttributeError):
            warn(
                "Unknown colour (%s); valid choices: [hex (#00ff00), %s]"
                % (value, ", ".join(self.COLOURS)),
                TldmWarning,
                stacklevel=2,
            )
            self._colour = None

    def __format__(self, format_spec: str) -> str:
        if format_spec:
            _type = format_spec[-1].lower()
            try:
                charset = {"a": self.ASCII, "u": self.UTF, "b": self.BLANK}[_type]
            except KeyError:
                charset = self.charset
            else:
                format_spec = format_spec[:-1]
            if format_spec:
                n_bars = int(format_spec)
                if n_bars < 0:
                    n_bars += self.default_len
            else:
                n_bars = self.default_len
        else:
            charset = self.charset
            n_bars = self.default_len

        nsyms = len(charset) - 1
        bar_length, frac_bar_length = divmod(int(self.frac * n_bars * nsyms), nsyms)

        res = charset[-1] * bar_length
        if bar_length < n_bars:  # whitespace padding
            res = res + charset[frac_bar_length] + charset[0] * (n_bars - bar_length - 1)
        return self.colour + res + self.COLOUR_RESET if self.colour else res


def format_sizeof(num: float, divisor: float = 1000) -> str:
    """
    Formats a number (>= 1) with SI Order of Magnitude prefixes.

    Parameters
    ----------
    num  : float
        Number (>= 1) to format.
    divisor  : float, optional
        Divisor between prefixes [default: 1000].

    Returns
    -------
    out  : str
        Number with Order of Magnitude SI unit postfix.
    """
    if num == 0:
        return "0"

    units = ["", "k", "M", "G", "T", "P", "E", "Z", "Y", "R", "Q"]

    # Calculate the exponent of the number's magnitude
    # This directly finds the correct unit without looping
    exponent = int(round(math.log(abs(num), divisor), 4))

    # Clamp the exponent to the number of available units to avoid an IndexError
    exponent = min(exponent, len(units) - 1)

    # Scale the number and select the correct unit
    num_scaled = num / (divisor**exponent)
    unit = units[exponent]

    # Apply the correct formatting based on the scaled number
    if abs(num_scaled) < 9.995:
        return f"{num_scaled:.2f}{unit}"
    elif abs(num_scaled) < 99.995:
        return f"{num_scaled:.1f}{unit}"
    else:
        return f"{num_scaled:.0f}{unit}"


def format_interval(t: float) -> str:
    """
    Formats a number of seconds as a clock time, [[Dd ]H:]MM:SS

    Parameters
    ----------
    t  : float
        Number of seconds.

    Returns
    -------
    out  : str
        [[Dd ]H:]MM:SS
    """
    mins, s = divmod(int(t), 60)
    h, m = divmod(mins, 60)
    days, h = divmod(h, 24)
    if days:
        return f"{days:d}d {h:d}:{m:02d}:{s:02d}"
    elif h:
        return f"{h:d}:{m:02d}:{s:02d}"
    else:
        return f"{m:02d}:{s:02d}"


def format_num(n: numbers.Real) -> str:
    """
    Intelligent scientific notation (.3g).

    Parameters
    ----------
    n  : int or float or Numeric
        A Number.

    Returns
    -------
    out  : str
        Formatted number.
    """
    f = f"{n:.3g}".replace("e+0", "e+").replace("e-0", "e-")
    n_str = str(n)
    return f if len(f) < len(n_str) else n_str


def get_status_printer(file: TextIO) -> Callable[[str], None]:
    """
    Manage the printing and in-place updating of a line of characters.
    Note that if the string is longer than a line, then in-place
    updating may not work (it will print a new line at each refresh).
    """
    fp = file
    fp_flush = getattr(fp, "flush", lambda: None)  # pragma: no cover
    if fp in (sys.stderr, sys.stdout):
        getattr(sys.stderr, "flush", lambda: None)()
        getattr(sys.stdout, "flush", lambda: None)()

    def fp_write(s):
        fp.write(str(s))
        fp_flush()

    last_len = 0

    def print_status(s: str) -> None:
        nonlocal last_len
        len_s = disp_len(s)
        fp_write("\r" + s + (" " * max(last_len - len_s, 0)))
        last_len = len_s

    return print_status


def format_meter(
    n: float | int,
    total: float | int | None,
    elapsed: float,
    ncols: int | None = None,
    prefix: str = "",
    ascii: bool | str = False,
    unit: str = "it",
    unit_scale: bool | int | float = False,
    rate: float | None = None,
    bar_format: str | None = None,
    postfix: str | None = None,
    unit_divisor: float = 1000,
    initial: float | int = 0,
    colour: str | None = None,
    title: bool = False,
    **extra_kwargs: dict[str, Any],
):
    """
    Return a string-based progress bar given some parameters

    Parameters
    ----------
    n  : int or float
        Number of finished iterations.
    total  : int or float or None
        The expected total number of iterations. If meaningless (None),
        only basic progress statistics are displayed (no ETA).
    elapsed  : float
        Number of seconds passed since start.
    ncols  : int, optional
        The width of the entire output message. If specified,
        dynamically resizes `{bar}` to stay within this bound
        [default: None]. If `0`, will not print any bar (only stats).
        The fallback is `{bar:10}`.
    prefix  : str, optional
        Prefix message (included in total width) [default: ''].
        Use as {desc} in bar_format string.
    ascii  : bool or str, optional
        If not set, use unicode (smooth blocks) to fill the meter
        [default: False]. The fallback is to use ASCII characters
        " 123456789#".
    unit  : str, optional
        The iteration unit [default: 'it'].
    unit_scale  : bool or int or float, optional
        If 1 or True, the number of iterations will be printed with an
        appropriate SI metric prefix (k = 10^3, M = 10^6, etc.)
        [default: False]. If any other non-zero number, will scale
        `total` and `n`.
    rate  : float, optional
        Manual override for iteration rate.
        If [default: None], uses n/elapsed.
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
    postfix  : *, optional
        Similar to `prefix`, but placed at the end
        (e.g. for additional stats).
        Note: postfix is usually a string (not a dict) for this method,
        and will if possible be set to postfix = ', ' + postfix.
        However other types are supported (#382).
    unit_divisor  : float, optional
        [default: 1000], ignored unless `unit_scale` is True.
    initial  : int or float, optional
        The initial counter value [default: 0].
    colour  : str, optional
        Bar colour (e.g. 'green', '#00ff00').

    Returns
    -------
    out  : Formatted meter and stats, ready to display.
    """

    # sanity check: total
    if total and n >= (total + 0.5):  # allow float imprecision (#849)
        total = None

    # apply custom scale if necessary
    if unit_scale and unit_scale not in (True, 1):
        if total:
            total *= unit_scale
        n *= unit_scale
        if rate:
            rate *= unit_scale  # by default rate = self.avg_dn / self.avg_dt
        unit_scale = False

    elapsed_str = format_interval(elapsed)

    # if unspecified, attempt to use rate = average speed
    # (we allow manual override since predicting time is an arcane art)
    if rate is None and elapsed:
        rate = (n - initial) / elapsed
    inv_rate = 1 / rate if rate else None

    rate_noinv_fmt = (
        ((format_sizeof(rate) if unit_scale else f"{rate:5.2f}") if rate else "?") + unit + "/s"
    )
    rate_inv_fmt = (
        ((format_sizeof(inv_rate) if unit_scale else f"{inv_rate:5.2f}") if inv_rate else "?")
        + "s/"
        + unit
    )
    rate_fmt = rate_inv_fmt if inv_rate and inv_rate > 1 else rate_noinv_fmt

    if unit_scale:
        n_fmt = format_sizeof(n, divisor=unit_divisor)
        total_fmt = format_sizeof(total, divisor=unit_divisor) if total is not None else "?"
    else:
        n_fmt = str(n)
        total_fmt = str(total) if total is not None else "?"

    with contextlib.suppress(TypeError):
        postfix = ", " + postfix if postfix else ""

    remaining = elapsed / (n - initial) * (total - n + initial) if rate and total else 0
    remaining_str = format_interval(remaining) if rate else "?"
    try:
        eta_dt = (
            datetime.now() + timedelta(seconds=remaining)
            if rate and total
            else datetime.fromtimestamp(0, UTC)
        )
    except OverflowError:
        eta_dt = datetime.max

    # format the stats displayed to the left and right sides of the bar
    if prefix:
        # old prefix setup work around
        bool_prefix_colon_already = prefix[-2:] == ": "
        l_bar = prefix if bool_prefix_colon_already else prefix + ": "
    else:
        l_bar = ""

    r_bar = f"| {n_fmt}/{total_fmt} [{elapsed_str}<{remaining_str}, {rate_fmt}{postfix}]"

    # Custom bar formatting
    # Populate a dict with all available progress indicators
    format_dict = {
        # slight extension of self.format_dict
        "n": n,
        "n_fmt": n_fmt,
        "total": total,
        "total_fmt": total_fmt,
        "elapsed": elapsed_str,
        "elapsed_s": elapsed,
        "ncols": ncols,
        "desc": prefix or "",
        "unit": unit,
        "rate": inv_rate if inv_rate and inv_rate > 1 else rate,
        "rate_fmt": rate_fmt,
        "rate_noinv": rate,
        "rate_noinv_fmt": rate_noinv_fmt,
        "rate_inv": inv_rate,
        "rate_inv_fmt": rate_inv_fmt,
        "postfix": postfix,
        "unit_divisor": unit_divisor,
        "colour": colour,
        # plus more useful definitions
        "remaining": remaining_str,
        "remaining_s": remaining,
        "l_bar": l_bar,
        "r_bar": r_bar,
        "eta": eta_dt,
        **extra_kwargs,
    }

    # total is known: we can predict some stats
    if total:
        # fractional and percentage progress
        frac = n / total
        percentage = frac * 100
        if percentage >= 99.5 and n != total:
            percentage = 99

        if title:
            OSC_PROGRESS = "\x1b]9;4;1;"
            OSC_END = "\7"
            l_bar += f"{OSC_PROGRESS}{round(percentage)}{OSC_END}{percentage:3.0f}%|"
        else:
            l_bar += f"{percentage:3.0f}%|"

        if ncols == 0:
            return l_bar[:-1] + r_bar[1:]

        format_dict.update(l_bar=l_bar)
        if bar_format:
            format_dict.update(percentage=percentage)

            # auto-remove colon for empty `{desc}`
            if not prefix:
                bar_format = bar_format.replace("{desc}: ", "")
        else:
            bar_format = "{l_bar}{bar}{r_bar}"

        bar_replacer = FormatReplace()
        nobar = bar_format.format(bar=bar_replacer, **format_dict)
        if not bar_replacer.format_called:
            return nobar  # no `{bar}`; nothing else to do

        # Formatting progress bar space available for bar's display
        full_bar = Bar(
            frac,
            max(1, ncols - disp_len(nobar)) if ncols else 10,
            charset=Bar.ASCII if ascii is True else ascii or Bar.UTF,
            colour=colour,
        )
        if not _is_ascii(full_bar.charset) and _is_ascii(bar_format):
            bar_format = str(bar_format)
        res = bar_format.format(bar=full_bar, **format_dict)
        return disp_trim(res, ncols) if ncols else res

    elif bar_format:
        # user-specified bar_format but no total
        l_bar += "|"
        format_dict.update(l_bar=l_bar, percentage=0)

        bar_replacer = FormatReplace()
        nobar = bar_format.format(bar=bar_replacer, **format_dict)
        if not bar_replacer.format_called:
            return nobar

        full_bar = Bar(
            0,
            max(1, ncols - disp_len(nobar)) if ncols else 10,
            charset=Bar.BLANK,
            colour=colour,
        )
        res = bar_format.format(bar=full_bar, **format_dict)
        return disp_trim(res, ncols) if ncols else res
    else:
        # no total: no progressbar, ETA, just progress stats
        return (
            f"{(prefix + ': ') if prefix else ''}"
            f"{n_fmt}{unit} [{elapsed_str}, {rate_fmt}{postfix}]"
        )


def _resize_signal_handler(signalnum, frame):
    """Handle terminal resize signal (SIGWINCH) to update dynamic ncols/nrows."""
    # Import here to avoid circular dependency
    from .std import tldm

    for cls in tldm.registered_classes:
        with cls.get_lock():
            for instance in cls._instances:
                if instance.dynamic_ncols_func:
                    ncols, nrows = instance.dynamic_ncols_func(instance.fp)
                    if not instance.keep_original_size[0]:
                        instance.ncols = ncols
                    if not instance.keep_original_size[1]:
                        instance.nrows = nrows
