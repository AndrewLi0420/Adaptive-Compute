"""Cooperative runtime: the workload throttles itself at safe points.

Generic throttling suspends the process with SIGSTOP wherever it happens to be
— possibly mid-GPU-command-buffer — and a controller that dies mid-stop leaves
the job frozen. A cooperative workload instead yields *between* units of work,
at boundaries it chose, and simply carries on if the controller disappears.

The yield arithmetic is a feedback-control problem, not a formula. For a region
that took `c` seconds, hitting a duty cycle of `f` means sleeping
`c * (1/f - 1)`. Applied naively that breaks in several ways, so:

* **Debt accounting.** Owed sleep accumulates in a debt counter and is only
  paid once it exceeds a minimum sleep. Thousands of tiny regions therefore
  produce occasional real sleeps rather than sub-millisecond ones that the OS
  cannot honour anyway. The minimum is deliberately large (50 ms) — see
  MIN_SLEEP_S, where the measurement is recorded.
* **The whole busy stretch is charged, not just the region.** Everything the
  loop does between regions still competes for the machine, and with short
  regions that overhead dominates.
* **Actual sleep is subtracted, not requested sleep.** Timer overshoot is real
  (measured at ~5 ms on this machine in M2), so paying down by what we actually
  slept lets the next cycle compensate instead of accumulating error.
* **Sleeps are capped**, so one very long region cannot produce a multi-minute
  stall; the remainder stays in the debt and is paid off over later regions.
* **Region duration is smoothed** (EMA) so a single slow step does not swing
  the yield.
* **fraction >= 1.0 is a zero-work path**: no sleeping, and the budget file is
  only stat-ed occasionally.

Concurrency: an AdaptiveRuntime instance is intended for one training loop and
is not thread-safe. The module-level `adaptive` singleton exists because that
is the ergonomic the spec asks for; all of its state lives on the instance.
"""

import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from adaptive_compute.sdk.channel import (
    Heartbeat,
    MetricsWriter,
    PublishedBudget,
    job_dir_from_env,
    read_budget,
)

log = logging.getLogger(__name__)

# Yield in chunks of at least this long. Tuned by measurement, not taste: with
# 5 ms sleeps a 0.25 budget delivered only 0.057 of unrestricted throughput
# despite hitting its duty cycle exactly, because chopping execution into
# millisecond slivers keeps the SoC clocked down and caches cold. Raising the
# minimum to 50 ms lifted that to 0.138 at the same duty, and left a 0.5 budget
# unaffected. Fewer, longer yields beat many short ones.
MIN_SLEEP_S = 0.05
MAX_SLEEP_S = 2.0  # one region can never stall the loop longer than this
BUDGET_POLL_S = 0.25  # how often to re-stat the budget file
PAUSE_SLICE_S = 0.2  # granularity of a cooperative pause
REGION_EMA_ALPHA = 0.3


class AdaptiveRuntime:
    """The object behind `from adaptive_compute import adaptive`.

    Outside a managed job (no job directory in the environment) every method is
    a cheap no-op, so an instrumented script runs unchanged on its own.
    """

    def __init__(self, job_dir: Path | None = None, env: dict[str, str] | None = None,
                 min_sleep_s: float = MIN_SLEEP_S):
        self._job_dir = job_dir if job_dir is not None else job_dir_from_env(env)
        self.min_sleep_s = min_sleep_s
        self._budget = PublishedBudget(written_at=time.time())
        self._budget_checked_at = 0.0
        self._heartbeat = Heartbeat(self._job_dir) if self._job_dir else None
        self._metrics = MetricsWriter(self._job_dir) if self._job_dir else None
        self._debt_s = 0.0
        self._busy_since: float | None = None
        self._region_ema_s: float | None = None
        self._regions = 0
        self._stale_logged = False

    # -- state -------------------------------------------------------------

    @property
    def active(self) -> bool:
        """True when running under a controller. False means every call is a no-op."""
        return self._job_dir is not None

    def current_budget(self) -> PublishedBudget:
        self._refresh_budget()
        return self._budget

    def recommended_batch_scale(self) -> float:
        """Advisory only: a hint the workload may choose to act on.

        Adaptive Compute deliberately does not resize batches itself — batch
        size changes training semantics, and doing that behind the workload's
        back would be both unsafe and algorithm-specific.
        """
        pressure = self.current_budget().memory_pressure
        if pressure == "critical":
            return 0.5
        if pressure == "warn":
            return 0.75
        return 1.0

    # -- the training-loop API --------------------------------------------

    @contextmanager
    def compute(self) -> Iterator[None]:
        """Wrap one unit of work; yields afterwards if the budget requires it."""
        if not self.active:
            yield
            return
        self.wait_while_paused()
        started = time.monotonic()
        try:
            yield
        finally:
            self._record_region(time.monotonic() - started)
            self._pay_debt()

    def yield_if_needed(self) -> None:
        """Pay down owed yield time outside a compute region.

        For loops with no clean region boundary, or extra yield points inside a
        long step.
        """
        if not self.active:
            return
        self.wait_while_paused()
        self._pay_debt()

    def report(self, **metrics: Any) -> None:
        """Record training metrics. Never raises into the training loop."""
        if not self.active or self._metrics is None:
            return
        payload = dict(metrics)
        payload.setdefault("ts", time.time())
        if self._region_ema_s is not None:
            payload.setdefault("step_time_ms", self._region_ema_s * 1000)
        self._metrics.append(payload)

    def wait_while_paused(self) -> None:
        """Block while the controller asks for a pause.

        Sleeps in slices, re-reading the budget each time, and gives up if the
        budget goes stale: a workload must never stay paused forever because
        the controller died.
        """
        if not self.active:
            return
        while True:
            budget = self.current_budget()
            if not budget.should_pause:
                return
            if budget.is_stale():
                self._log_stale()
                return
            self._beat()
            time.sleep(PAUSE_SLICE_S)
            self._budget_checked_at = 0.0  # force a re-read on the next check

    # -- internals ---------------------------------------------------------

    def _refresh_budget(self) -> None:
        now = time.monotonic()
        if now - self._budget_checked_at < BUDGET_POLL_S:
            return
        self._budget_checked_at = now
        if self._job_dir is None:
            return
        fresh = read_budget(self._job_dir / "budget.json")
        if fresh is not None:
            self._budget = fresh
            self._stale_logged = False
        elif self._budget.is_stale():
            self._log_stale()

    def _log_stale(self) -> None:
        if not self._stale_logged:
            self._stale_logged = True
            log.warning(
                "adaptive_compute: no budget update for %.0fs; continuing at full speed",
                self._budget.age_s,
            )

    def _record_region(self, duration_s: float) -> None:
        self._regions += 1
        if self._region_ema_s is None:
            self._region_ema_s = duration_s
        else:
            self._region_ema_s = (
                REGION_EMA_ALPHA * duration_s + (1 - REGION_EMA_ALPHA) * self._region_ema_s
            )
        self._beat()

    def _accrue_debt(self, now: float) -> None:
        """Charge for every non-sleeping second, not just the region itself.

        Measuring only the compute region undercharges: the loop's own work
        between regions — reporting metrics, fetching the next batch, and this
        SDK's overhead — is time the workload is competing for the machine. With
        short regions that gap dominates, and the achieved duty cycle overshoots
        the budget badly (measured 0.63 against a 0.50 request before this
        change). Charging for the whole busy stretch is what makes
        compute_fraction mean "share of wall time spent not yielding".
        """
        busy = now - (self._busy_since if self._busy_since is not None else now)
        self._busy_since = now
        budget = self.current_budget()
        if budget.is_stale():
            self._log_stale()
            return
        fraction = min(1.0, max(0.01, budget.compute_fraction))
        if fraction >= 1.0:
            return
        self._debt_s += busy * (1.0 / fraction - 1.0)

    def _pay_debt(self) -> None:
        self._accrue_debt(time.monotonic())
        if self._debt_s < self.min_sleep_s:
            return
        requested = min(self._debt_s, MAX_SLEEP_S)
        started = time.monotonic()
        time.sleep(requested)
        ended = time.monotonic()
        # Subtract what we actually slept, not what we asked for: timer
        # overshoot then corrects itself on the next region rather than
        # accumulating into a systematic slowdown.
        self._debt_s -= ended - started
        # A little credit is allowed (it absorbs overshoot); unbounded credit is
        # not, or a budget change could be ignored for a long time.
        self._debt_s = max(self._debt_s, -MAX_SLEEP_S)
        self._busy_since = ended

    def _beat(self) -> None:
        if self._heartbeat is not None:
            self._heartbeat.beat(self._regions)

    # -- test/support ------------------------------------------------------

    @property
    def debt_s(self) -> float:
        return self._debt_s

    @property
    def regions(self) -> int:
        return self._regions


def _default_runtime() -> AdaptiveRuntime:
    runtime = AdaptiveRuntime()
    if runtime.active:
        runtime._beat()  # advertise immediately so the controller stops suspending us
        log.debug("adaptive_compute cooperative runtime active (pid %d)", os.getpid())
    return runtime


adaptive = _default_runtime()
