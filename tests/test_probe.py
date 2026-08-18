import time

import psutil
import pytest

from adaptive_compute.monitor.probe import (
    ProbeStats,
    ResponsivenessProbe,
    parse_sample_line,
    percentile,
)


def test_percentile_nearest_rank():
    values = [float(i) for i in range(1, 101)]
    assert percentile(values, 50) == 50.0
    assert percentile(values, 95) == 95.0
    assert percentile(values, 99) == 99.0
    assert percentile(values, 100) == 100.0


def test_percentile_small_samples():
    assert percentile([7.0], 95) == 7.0
    assert percentile([1.0, 2.0], 50) == 1.0
    assert percentile([2.0, 1.0], 95) == 2.0  # unsorted input


def test_percentile_empty_raises():
    with pytest.raises(ValueError):
        percentile([], 50)


def test_parse_sample_line():
    assert parse_sample_line("1700000000.500 4.250 0.310\n") == (1700000000.5, 4.25, 0.31)


@pytest.mark.parametrize("line", ["", "\n", "garbage", "1.0 2.0", "1.0 2.0 3.0 4.0", "a b c"])
def test_parse_sample_line_rejects_bad_input(line):
    assert parse_sample_line(line) is None


def _fill(probe: ResponsivenessProbe, count: int, now: float, wake=5.0, rtt=0.15) -> None:
    """Append `count` samples ending at `now`, oldest first (as the worker does)."""
    for i in reversed(range(count)):
        probe._samples.append((now - i * probe.interval_s, wake, rtt))


def test_stats_needs_min_samples():
    probe = ResponsivenessProbe(min_samples=20)
    now = time.time()
    _fill(probe, 19, now)
    assert probe.stats(now=now) is None
    _fill(probe, 1, now)
    assert probe.stats(now=now) is not None


def test_stats_reports_rtt_as_headline_and_wake_separately():
    """The 5 ms timer floor must not be folded into the round-trip signal."""
    probe = ResponsivenessProbe()
    now = time.time()
    _fill(probe, 50, now, wake=5.0, rtt=0.5)
    stats = probe.stats(now=now)
    assert isinstance(stats, ProbeStats)
    assert stats.p50_ms == pytest.approx(0.5)
    assert stats.p95_ms == pytest.approx(0.5)
    assert stats.wake_p95_ms == pytest.approx(5.0)
    assert stats.sample_count == 50


def test_stats_excludes_samples_outside_window():
    probe = ResponsivenessProbe(window_s=10.0, min_samples=5)
    now = time.time()
    for i in reversed(range(30)):
        probe._samples.append((now - i, 4.0, 0.3))  # one per second, 30s back
    stats = probe.stats(now=now)
    assert stats.sample_count == 11  # now-0 .. now-10 inclusive


def test_stats_none_when_stale():
    probe = ResponsivenessProbe(min_samples=5)
    now = time.time()
    _fill(probe, 50, now - 60)  # last sample a minute old
    assert probe.stats(now=now) is None


def test_stats_none_when_empty():
    assert ResponsivenessProbe().stats() is None


def test_sample_returns_no_fields_when_unavailable():
    """Provider protocol: unavailable means absent, never a fabricated value."""
    assert ResponsivenessProbe().sample() == {}


def test_sample_reports_percentiles():
    probe = ResponsivenessProbe()
    now = time.time()
    _fill(probe, 50, now, wake=5.0, rtt=1.25)
    fields = probe.sample()
    assert fields["responsiveness_latency_ms"] == pytest.approx(1.25)
    assert fields["responsiveness_p50_ms"] == pytest.approx(1.25)
    assert set(fields) == {
        "responsiveness_latency_ms",
        "responsiveness_p50_ms",
        "responsiveness_p99_ms",
    }


def test_double_start_raises():
    probe = ResponsivenessProbe()
    with probe:
        with pytest.raises(RuntimeError):
            probe.start()


def test_stop_without_start_is_safe():
    ResponsivenessProbe().stop()


@pytest.mark.slow
def test_worker_produces_real_samples():
    with ResponsivenessProbe(interval_s=0.05, min_samples=10) as probe:
        assert probe.is_alive()
        time.sleep(2.0)
        stats = probe.stats()
    assert stats is not None
    assert stats.sample_count >= 10
    # Wide bounds: this only asserts the probe measures something plausible,
    # not that the machine running the tests is quiet.
    assert 0 < stats.p50_ms < 500
    assert 0 < stats.wake_p95_ms < 500


@pytest.mark.slow
def test_stop_leaves_no_orphan_partner():
    """The worker spawns its own echo partner; neither may outlive stop()."""
    probe = ResponsivenessProbe(interval_s=0.05)
    probe.start()
    worker = psutil.Process(probe._proc.pid)
    time.sleep(1.0)
    partners = worker.children()
    assert len(partners) == 1
    probe.stop()
    time.sleep(0.5)
    assert not any(p.is_running() and p.status() != psutil.STATUS_ZOMBIE for p in partners)


@pytest.mark.slow
def test_worker_death_makes_stats_unavailable():
    probe = ResponsivenessProbe(interval_s=0.05, min_samples=5)
    probe.start()
    time.sleep(1.0)
    assert probe.stats() is not None
    partner = psutil.Process(probe._proc.pid).children()[0]

    probe._proc.kill()  # SIGKILL: the worker's cleanup never runs
    probe._proc.wait(timeout=2)
    time.sleep(3.2)  # exceed STALE_AFTER_S

    assert probe.stats() is None
    assert probe.sample() == {}
    # the partner must still exit on its own, via EOF on the closed pipe
    assert not (partner.is_running() and partner.status() != psutil.STATUS_ZOMBIE)
    probe.stop()
