import atexit
from collections.abc import Callable
from threading import Event, Thread, current_thread
from time import time
from typing import TYPE_CHECKING
from warnings import warn

if TYPE_CHECKING:
    from tldm.std import tldm


class TldmSynchronisationWarning(RuntimeWarning):
    """tldm multi-thread/-process errors which may cause incorrect nesting
    but otherwise no adverse effects"""

    pass


class TMonitor(Thread):
    """
    Monitoring thread for tldm bars.
    Monitors if tldm bars are taking too much time to display
    and readjusts miniters automatically if necessary.

    Parameters
    ----------
    tldm_cls  : class
        tldm class to use (can be core tldm or a submodule).
    sleep_interval  : float
        Time to sleep between monitoring checks.
    """

    _test: dict[str, Event] = {}  # internal vars for unit testing

    name: str
    daemon: bool
    woken: float
    tldm_cls: type["tldm"]
    sleep_interval: float
    _time: Callable[[], float]
    was_killed: Event

    def __init__(self, tldm_cls: type["tldm"], sleep_interval: float) -> None:
        Thread.__init__(self)
        self.name = "tldm_monitor"
        self.daemon = True  # kill thread when main killed (KeyboardInterrupt)
        self.woken = 0  # last time woken up, to sync with monitor
        self.tldm_cls = tldm_cls
        self.sleep_interval = sleep_interval
        self._time = self._test.get("time", time)  # type: ignore[assignment]
        self.was_killed = self._test.get("Event", Event)()  # type: ignore[operator]
        atexit.register(self.exit)
        self.start()

    def exit(self) -> bool:
        self.was_killed.set()
        if self is not current_thread():
            self.join()
        return self.report()

    def get_instances(self) -> list["tldm"]:
        # returns a copy of started `tldm_cls` instances
        return [
            i
            for i in self.tldm_cls._instances.copy()
            # Avoid race by checking that the instance started
            if hasattr(i, "start_t")
        ]

    def run(self) -> None:
        cur_t = self._time()
        while True:
            # After processing and before sleeping, notify that we woke
            # Need to be done just before sleeping
            self.woken = cur_t
            # Sleep some time...
            self.was_killed.wait(self.sleep_interval)
            # Quit if killed
            if self.was_killed.is_set():
                return
            # Then monitor!
            # Acquire lock (to access _instances)
            with self.tldm_cls.get_lock():
                cur_t = self._time()
                # Check tldm instances are waiting too long to print
                instances = self.get_instances()
                for instance in instances:
                    # Check event in loop to reduce blocking time on exit
                    if self.was_killed.is_set():
                        return
                    # Only if mininterval > 1 (else iterations are just slow)
                    # and last refresh exceeded maxinterval
                    if (
                        instance.miniters > 1
                        and (cur_t - instance.last_print_t) >= instance.maxinterval
                    ):
                        # force bypassing miniters on next iteration
                        # (dynamic_miniters adjusts mininterval automatically)
                        instance.miniters = 1
                        # Refresh now! (works only for manual tldm)
                        instance.refresh(nolock=True)
                    # Remove accidental long-lived strong reference
                    del instance
                if instances != self.get_instances():  # pragma: nocover
                    warn(
                        "Set changed size during iteration"
                        + " (see https://github.com/tldm/tldm/issues/481)",
                        TldmSynchronisationWarning,
                        stacklevel=2,
                    )
                # Remove accidental long-lived strong references
                del instances

    def report(self) -> bool:
        return not self.was_killed.is_set()
