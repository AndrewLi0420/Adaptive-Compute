"""Approximate a compute fraction for an arbitrary process.

Adaptive Compute does not cap CPU or GPU utilization directly — macOS exposes
no such control to an unprivileged process. For generic (non-SDK) workloads it
suspends and resumes the process group so that, over each period, the workload
runs for roughly `compute_fraction` of the wall clock. That is an
approximation, not a quota, and it is coarse:

* resolution is one tick (default 50 ms), so fractions are quantized;
* a suspended process still holds all of its memory, so this reduces CPU
  contention but not memory pressure;
* SIGSTOP freezes the process wherever it happens to be, including mid-GPU
  command buffer, which is why cooperative yielding (M6) is the better path
  for workloads that can integrate with the SDK.

The duty cycler is a pure state machine evaluated from the control loop: it
never sleeps and never blocks, so a shutdown request is always serviced
promptly (the lesson from M3's escalation bug).
"""

import logging

from adaptive_compute.process.job import JobState
from adaptive_compute.process.manager import JobManager
from adaptive_compute.scheduler.policy import MAX_COMPUTE_FRACTION, ResourceBudget

log = logging.getLogger(__name__)

DEFAULT_PERIOD_S = 1.0
DEFAULT_MIN_SLICE_S = 0.05


class DutyCycler:
    """Decides, for a point in time, whether the workload should be running."""

    def __init__(self, period_s: float = DEFAULT_PERIOD_S, min_slice_s: float = DEFAULT_MIN_SLICE_S):
        self.period_s = period_s
        self.min_slice_s = min_slice_s
        self._cycle_start: float | None = None

    def should_run(self, fraction: float, now: float) -> bool:
        # Slices shorter than min_slice would mean signalling the process more
        # often than it can usefully make progress, so collapse to "always on"
        # or "always off" rather than thrashing SIGSTOP/SIGCONT.
        run_slice = fraction * self.period_s
        if run_slice >= self.period_s - self.min_slice_s:
            self._cycle_start = None
            return True
        if run_slice < self.min_slice_s:
            run_slice = self.min_slice_s

        if self._cycle_start is None:
            self._cycle_start = now
        elapsed = now - self._cycle_start
        while elapsed >= self.period_s:
            self._cycle_start += self.period_s
            elapsed -= self.period_s
        return elapsed < run_slice

    def reset(self) -> None:
        self._cycle_start = None


class ProcessThrottler:
    """Applies a ResourceBudget to a managed process group.

    Concurrency: single-threaded, driven from the same control loop as the
    sampler and the job manager.
    """

    def __init__(self, manager: JobManager, period_s: float = DEFAULT_PERIOD_S):
        self.manager = manager
        self.cycler = DutyCycler(period_s=period_s)

    def apply(self, budget: ResourceBudget, now: float) -> None:
        job = self.manager.job
        if job.state.is_terminal:
            return

        if budget.should_pause:
            self.cycler.reset()
            self.manager.set_suspended(True)
            self.manager.set_state(JobState.PAUSED)
            return

        if budget.compute_fraction >= MAX_COMPUTE_FRACTION:
            self.cycler.reset()
            self.manager.set_suspended(False)
            self.manager.set_state(JobState.RUNNING)
            return

        running = self.cycler.should_run(budget.compute_fraction, now)
        self.manager.set_suspended(not running)
        self.manager.set_state(JobState.THROTTLED)
