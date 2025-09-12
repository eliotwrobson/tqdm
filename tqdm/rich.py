"""
`rich.progress` decorator for iterators.

Usage:
>>> from tqdm.rich import trange, tqdm
>>> for i in trange(10):
...     ...
"""

from contextlib import nullcontext
from typing import Any, Iterable, cast
from warnings import warn
from tqdm.utils import format_interval, format_sizeof

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    Table,
    Task,
    Text,
    TimeRemainingColumn,
)

from tqdm.std import TqdmWarning
from tqdm.std import tqdm as std_tqdm

RenderReturnType = Text | str

__author__ = {"github.com/": ["casperdcl"]}
__all__ = ["tqdm_rich", "trrange", "tqdm", "trange"]


class UnitScaleColumn(ProgressColumn):
    def __init__(self, unit_scale: bool = False, unit_divisor: int = 1000) -> None:
        self.unit_scale = unit_scale
        self.unit_divisor = unit_divisor
        super().__init__()

    def unit_format(self, task: Task, num: float, fmt: str = "") -> str:
        if task.fields["unit_scale"]:
            return format_sizeof(num, divisor=task.fields["unit_divisor"])
        return f"{num:{fmt}}"


class FractionColumn(UnitScaleColumn):
    def render(self, task: Task) -> RenderReturnType:
        has_total = task.total is not None

        if has_total:
            n_fmt = self.unit_format(task, cast(float, task.completed))
            total_fmt = self.unit_format(task, cast(float, task.total))
            return Text(f"{n_fmt}/{total_fmt}", style="progress.download")

        return ""


class RateColumn(UnitScaleColumn):
    """Renders human readable transfer speed."""

    def __init__(
        self, unit: str = "it", unit_scale: bool = False, unit_divisor: int = 1000
    ) -> None:
        super().__init__(unit_scale=unit_scale, unit_divisor=unit_divisor)
        self.unit = unit

    def render(self, task: Task) -> Text | str:
        """Show data transfer speed."""
        speed = task.fields["rate"]
        if task.fields["elapsed"] and speed is None:
            speed = task.completed / task.fields["elapsed"]
        if speed is not None:
            inv_speed = 1 / speed if speed != 0 else None
            if inv_speed and inv_speed > 1:
                unit_fmt = f"s/{task.fields['unit']}"
                speed_ = inv_speed
            else:
                unit_fmt = f"{task.fields['unit']}/s"
                speed_ = speed

            speed_fmt = self.unit_format(task, speed_, fmt="5.2f")
        else:
            speed_fmt = "?"
            unit_fmt = f"{task.fields['unit']}/s"

        return Text(f"{speed_fmt}{unit_fmt}", style="progress.data.speed")


class UnitCompletedColumn(UnitScaleColumn):
    def render(self, task: Task) -> RenderReturnType:
        if task.total is None:
            completed = self.unit_format(task, task.completed)
            return Text(
                f"{completed:>3}{task.fields['unit']}", style="progress.percentage"
            )
        else:
            return Text(f"{task.percentage:>3.0f}%", style="progress.percentage")


class CompactTimeElapsedColumn(ProgressColumn):
    def render(self, task: Task) -> RenderReturnType:
        elapsed = task.fields["elapsed"]
        formatted = format_interval(int(elapsed)) if elapsed else "--:--"
        return Text(formatted, style="progress.elapsed")


class PrefixTimeRemainingColumn(TimeRemainingColumn):
    def __init__(self, prefix_str: str = "<", *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.prefix_txt = Text(prefix_str)

    def render(self, task: Task) -> RenderReturnType:  # type: ignore[override]
        return (
            (self.prefix_txt + super().render(task)) if task.total is not None else ""
        )


class PostFixColumn(ProgressColumn):
    def render(self, task: Task) -> RenderReturnType:
        postfix = task.fields.get("postfix")
        return Text(f", {postfix}", style="progress.percentage") if postfix else ""


class NoPaddingProgress(Progress):
    def make_tasks_table(self, tasks: Iterable[Task]) -> Table:
        table = super().make_tasks_table(tasks)
        table.padding = 0
        return table


class ASCIIConsole(Console):
    @property
    def encoding(self) -> str:
        return "ascii"


class tqdm_rich(std_tqdm):  # pragma: no cover
    """Experimental rich.progress GUI version of tqdm!"""

    # TODO: @classmethod: write()?
    _progress: Progress | None

    def __new__(cls, *_, **__):
        return object.__new__(cls)

    @staticmethod
    def _get_free_pos(*_, **__):
        pass

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        This class accepts the following parameters *in addition* to
        the parameters accepted by `tqdm`.

        Parameters
        ----------
        progress  : tuple, optional
            arguments for `rich.progress.Progress()`.
        options  : dict, optional
            keyword arguments for `rich.progress.Progress()`.
        bar_options : dict, optional
            keyword arguments for `rich.progress.BarColumn()`.
        """
        kwargs = kwargs.copy()
        kwargs["gui"] = True

        progress_columns = kwargs.pop("progress", None)
        options = kwargs.pop("options", {}).copy()
        bar_options = kwargs.pop("bar_options", {}).copy()
        for k in ("position", "bar_format"):
            if kwargs.pop(k, None) is not None:
                warn(
                    f"tqdm.rich does not support the `{k}` option. ",
                    TqdmWarning,
                    stacklevel=2,
                )

        # NOTE: temporary dummy_lock to reuse std_tqdm's __init__
        self._lock = nullcontext()  # type: ignore[misc, assignment]
        self._instances = [self]  # type: ignore[misc, assignment]
        super().__init__(*args, **kwargs)
        del self._lock
        del self._instances

        if self.disable:
            return

        d = self.format_dict
        if progress_columns is None:
            description = (
                "[progress.description]{task.description}: " if self.desc else ""
            )
            completed = UnitCompletedColumn(
                unit_scale=d["unit_scale"], unit_divisor=d["unit_divisor"]
            )
            bar_options.setdefault("bar_width", None)
            if d["colour"] is not None:
                bar_options.setdefault("complete_style", d["colour"])
                bar_options.setdefault("finished_style", d["colour"])
                bar_options.setdefault("pulse_style", d["colour"])
            progress_columns = (
                description,
                completed,
                " ",
                BarColumn(**bar_options),
                " ",
                FractionColumn(
                    unit_scale=d["unit_scale"], unit_divisor=d["unit_divisor"]
                ),
                " [",
                CompactTimeElapsedColumn(),
                PrefixTimeRemainingColumn(compact=True),
                ", ",
                RateColumn(
                    unit=d["unit"],
                    unit_scale=d["unit_scale"],
                    unit_divisor=d["unit_divisor"],
                ),
                PostFixColumn(),
                "]",
            )

        cls = self.__class__
        if not hasattr(cls, "_progress") or cls._progress is None:
            options.setdefault("transient", not self.leave)

            if options.get("console") is not None:
                console: Console = options["console"]
                console.file = self.fp
                console.width = d["ncols"]
                console.height = d["nrows"]
                if d["ascii"] and not console.encoding == "ascii":
                    warn(
                        "ascii output requested but passed console's encoding is not 'ascii'. "
                        "See `tqdm.rich.ASCIIConsole` to force ASCII rendering.",
                        TqdmWarning,
                        stacklevel=2,
                    )
            else:
                console_cls = ASCIIConsole if d["ascii"] else Console
                options["console"] = console_cls(
                    width=d["ncols"], height=d["nrows"], file=self.fp
                )

            cls._progress = NoPaddingProgress(*progress_columns, **options)
            cls._progress.__enter__()
        else:
            if options.get("console") is not None:
                warn(
                    "ignoring passed `console` since tqdm_rich._progress exists",
                    TqdmWarning,
                    stacklevel=2,
                )

            if kwargs.get("ascii") is True and not isinstance(
                cls._progress.console, ASCIIConsole
            ):
                warn(
                    "ascii=True but global console is not ASCIIConsole. "
                    "Using (non-ascii) global console.",
                    TqdmWarning,
                    stacklevel=2,
                )

        with cls._progress._lock:
            # workaround to not refresh on task addition
            _disable, cls._progress.disable = cls._progress.disable, True
            task_id = cls._progress.add_task(self.desc or "", **d, start=False)
            self._task = next(t for t in cls._progress.tasks if t.id == task_id)
            cls._progress.disable = _disable

    def close(self) -> None:
        if self.disable:
            return
        cls = self.__class__

        if cls._progress is None or self._task is None:
            return
        with cls._progress._lock:
            if not self._task.finished:
                cls._progress.stop_task(self._task.id)
                self._task.finished_time = self._task.stop_time
            if not self.leave:
                self._task.visible = False
            self.display(
                refresh=cls._progress.console.is_jupyter
            )  # print 100%, vis #1306
            if all(t.finished for t in cls._progress.tasks):
                cls._progress.__exit__(None, None, None)
                cls._progress = None

    def clear(self, *_, **__) -> None:
        pass

    def display(self, refresh: bool = False, *_, **__) -> None:  # type: ignore[override]
        cls = self.__class__
        if not hasattr(cls, "_progress") or cls._progress is None or self._task is None:
            return
        if not self._task.started and self.n > 0:
            cls._progress.start_task(self._task.id)

        d = self.format_dict
        self._task.fields["rate"] = d["rate"]
        self._task.fields["postfix"] = d["postfix"]

        cls._progress.update(
            self._task.id,
            completed=self.n,
            description=self.desc,
            refresh=refresh,
            **d,
        )

        cls._progress.console.width = d["ncols"]
        cls._progress.console.height = d["nrows"]
        cls._progress.console.file = self.fp

    def refresh(self, *_, **__) -> None:
        self.display()

    def reset(self, total: int | None = None) -> None:
        """
        Resets to 0 iterations for repeated use.

        Parameters
        ----------
        total  : int or float, optional. Total to use for the new bar.
        """
        cls = self.__class__
        if cls._progress is not None:
            cls._progress.reset(task_id=self._task.id)  # see #1378
        super().reset(total=total)


def trrange(*args, **kwargs):
    """Shortcut for `tqdm.rich.tqdm(range(*args), **kwargs)`."""
    return tqdm_rich(range(*args), **kwargs)


# Aliases
tqdm = tqdm_rich
trange = trrange
