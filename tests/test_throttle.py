import sys
import time

import psutil
import pytest

from adaptive_compute.process import JobManager, JobState
from adaptive_compute.process.throttle import DutyCycler, ProcessThrottler
from adaptive_compute.scheduler.policy import ResourceBudget


# -- duty cycler: a pure state machine, so time is supplied ----------------


def test_full_fraction_always_runs():
    cycler = DutyCycler(period_s=1.0)
    assert all(cycler.should_run(1.0, t / 10) for t in range(30))


def test_half_fraction_runs_half_of_each_period():
    cycler = DutyCycler(period_s=1.0, min_slice_s=0.05)
    ticks = [cycler.should_run(0.5, t * 0.05) for t in range(60)]  # 3 periods
    assert sum(ticks) == pytest.approx(len(ticks) / 2, abs=3)


def test_duty_ratio_tracks_the_requested_fraction():
    for fraction in (0.25, 0.5, 0.75):
        cycler = DutyCycler(period_s=1.0, min_slice_s=0.05)
        ticks = [cycler.should_run(fraction, t * 0.01) for t in range(1000)]
        assert sum(ticks) / len(ticks) == pytest.approx(fraction, abs=0.02)


def test_cycle_repeats_across_periods():
    cycler = DutyCycler(period_s=1.0, min_slice_s=0.05)
    assert cycler.should_run(0.5, 0.0)
    assert not cycler.should_run(0.5, 0.7)
    assert cycler.should_run(0.5, 1.2)  # next period, back in the run slice
    assert not cycler.should_run(0.5, 1.8)


def test_gaps_in_time_do_not_break_the_cycle():
    """The control loop can stall; the cycler must resynchronise, not spin."""
    cycler = DutyCycler(period_s=1.0, min_slice_s=0.05)
    cycler.should_run(0.5, 0.0)
    assert cycler.should_run(0.5, 100.2) in (True, False)  # must simply not hang
    assert cycler.should_run(0.5, 100.9) is False


def test_tiny_slices_collapse_instead_of_thrashing():
    """Below one slice, signalling faster than the process can progress is worse
    than not throttling at all."""
    cycler = DutyCycler(period_s=1.0, min_slice_s=0.1)
    assert cycler.should_run(0.99, 0.0) is True  # stop slice too short
    ticks = [cycler.should_run(0.01, t * 0.01) for t in range(300)]
    assert sum(ticks) / len(ticks) == pytest.approx(0.1, abs=0.02)  # floored at min_slice


# -- throttler against a real process --------------------------------------

pytestmark_slow = pytest.mark.slow


def spinner(tmp_path) -> JobManager:
    return JobManager([sys.executable, "-c", "\nwhile True: sum(i*i for i in range(1000))"],
                      root=tmp_path)


@pytest.mark.slow
def test_full_budget_leaves_process_running(tmp_path):
    m = spinner(tmp_path)
    m.start()
    try:
        throttler = ProcessThrottler(m)
        throttler.apply(ResourceBudget(compute_fraction=1.0), time.monotonic())
        assert not m.suspended
        assert m.job.state is JobState.RUNNING
    finally:
        m.terminate(grace_s=2)


@pytest.mark.slow
def test_pause_budget_suspends_the_process(tmp_path):
    m = spinner(tmp_path)
    m.start()
    try:
        throttler = ProcessThrottler(m)
        throttler.apply(ResourceBudget(should_pause=True), time.monotonic())
        assert m.suspended
        assert m.job.state is JobState.PAUSED
        assert psutil.Process(m.job.pid).status() == psutil.STATUS_STOPPED

        throttler.apply(ResourceBudget(compute_fraction=1.0), time.monotonic())
        assert not m.suspended
        assert m.job.state is JobState.RUNNING
    finally:
        m.terminate(grace_s=2)


@pytest.mark.slow
def test_partial_budget_marks_throttled_and_toggles(tmp_path):
    m = spinner(tmp_path)
    m.start()
    try:
        throttler = ProcessThrottler(m, period_s=0.4)
        seen = set()
        start = time.monotonic()
        while time.monotonic() - start < 1.5:
            throttler.apply(ResourceBudget(compute_fraction=0.5), time.monotonic())
            seen.add(m.suspended)
            time.sleep(0.02)
        assert seen == {True, False}  # actually duty cycling, not stuck
        assert m.job.state is JobState.THROTTLED
    finally:
        m.terminate(grace_s=2)


@pytest.mark.slow
def test_terminate_wakes_a_duty_cycled_process(tmp_path):
    """A THROTTLED job is stopped for part of every period; shutdown must
    still work when the signal lands in a stop phase."""
    m = spinner(tmp_path)
    m.start()
    throttler = ProcessThrottler(m)
    throttler.apply(ResourceBudget(should_pause=True), time.monotonic())
    assert m.suspended
    assert m.terminate(grace_s=5) is JobState.STOPPED


@pytest.mark.slow
def test_throttling_a_finished_job_is_a_noop(tmp_path):
    m = JobManager([sys.executable, "-c", "pass"], root=tmp_path)
    m.start()
    m.wait(timeout=10)
    ProcessThrottler(m).apply(ResourceBudget(should_pause=True), time.monotonic())
    assert m.job.state is JobState.COMPLETED
    assert not m.suspended
