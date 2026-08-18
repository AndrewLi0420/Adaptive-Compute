import json
import time

import pytest

from adaptive_compute.sdk.channel import (
    BUDGET_STALE_AFTER_S,
    BudgetPublisher,
    Heartbeat,
    MetricsTailer,
    MetricsWriter,
    cooperative_is_active,
    read_budget,
)
from adaptive_compute.sdk.runtime import AdaptiveRuntime


def publish(job_dir, fraction=1.0, should_pause=False, age_s=0.0, **extra):
    payload = {
        "compute_fraction": fraction,
        "should_pause": should_pause,
        "written_at": time.time() - age_s,
        "seq": 1,
        **extra,
    }
    (job_dir / "budget.json").write_text(json.dumps(payload))


# -- channel ---------------------------------------------------------------


def test_budget_round_trip(tmp_path):
    BudgetPublisher(tmp_path).publish(0.4, False, memory_pressure="warn", mode="INTERACTIVE")
    budget = read_budget(tmp_path / "budget.json")
    assert budget.compute_fraction == 0.4
    assert budget.should_pause is False
    assert budget.memory_pressure == "warn"
    assert budget.mode == "INTERACTIVE"
    assert budget.seq == 1


def test_publisher_increments_seq(tmp_path):
    publisher = BudgetPublisher(tmp_path)
    publisher.publish(1.0, False)
    publisher.publish(0.5, False)
    assert read_budget(tmp_path / "budget.json").seq == 2


def test_missing_or_corrupt_budget_reads_as_none(tmp_path):
    assert read_budget(tmp_path / "absent.json") is None
    bad = tmp_path / "budget.json"
    bad.write_text("{not json")
    assert read_budget(bad) is None
    bad.write_text('{"unexpected": true}')
    assert read_budget(bad) is None


def test_staleness(tmp_path):
    publish(tmp_path, age_s=0)
    assert not read_budget(tmp_path / "budget.json").is_stale()
    publish(tmp_path, age_s=BUDGET_STALE_AFTER_S + 5)
    assert read_budget(tmp_path / "budget.json").is_stale()


def test_heartbeat_marks_cooperative(tmp_path):
    assert not cooperative_is_active(tmp_path)
    Heartbeat(tmp_path).beat(regions=1, force=True)
    assert cooperative_is_active(tmp_path)


def test_stale_heartbeat_is_not_cooperative(tmp_path):
    Heartbeat(tmp_path).beat(regions=1, force=True)
    assert not cooperative_is_active(tmp_path, now=time.time() + 3600)


def test_heartbeat_is_rate_limited(tmp_path):
    heartbeat = Heartbeat(tmp_path, interval_s=60)
    heartbeat.beat(regions=1, force=True)
    first = json.loads((tmp_path / "sdk.json").read_text())["regions"]
    heartbeat.beat(regions=99)  # too soon; must not rewrite
    assert json.loads((tmp_path / "sdk.json").read_text())["regions"] == first


def test_metrics_writer_and_tailer(tmp_path):
    writer = MetricsWriter(tmp_path, flush_interval_s=0)
    tailer = MetricsTailer(tmp_path)
    writer.append({"step": 1, "loss": 2.0})
    writer.flush()
    assert tailer.read_new() == [{"step": 1, "loss": 2.0}]
    assert tailer.read_new() == []  # nothing new the second time

    writer.append({"step": 2})
    writer.flush()
    assert tailer.read_new() == [{"step": 2}]


def test_tailer_waits_for_complete_lines(tmp_path):
    """A partially flushed write must not be parsed as garbage."""
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"step": 1}\n{"step": 2')  # second line incomplete
    tailer = MetricsTailer(tmp_path)
    assert tailer.read_new() == [{"step": 1}]
    with path.open("a") as handle:
        handle.write("}\n")
    assert tailer.read_new() == [{"step": 2}]


def test_writer_buffers_until_flush(tmp_path):
    writer = MetricsWriter(tmp_path, flush_interval_s=60, max_buffered=1000)
    for i in range(10):
        writer.append({"step": i})
    assert not (tmp_path / "metrics.jsonl").exists()
    writer.flush()
    assert len(MetricsTailer(tmp_path).read_new()) == 10


def test_unserialisable_metrics_are_dropped_not_raised(tmp_path):
    writer = MetricsWriter(tmp_path, flush_interval_s=0)
    writer.append({"bad": object()})  # must not raise
    writer.append({"good": 1})
    writer.flush()
    assert MetricsTailer(tmp_path).read_new() == [{"good": 1}]


# -- runtime ---------------------------------------------------------------


def test_inactive_outside_a_managed_job(tmp_path):
    runtime = AdaptiveRuntime(env={})
    assert not runtime.active
    with runtime.compute():
        pass
    runtime.yield_if_needed()
    runtime.report(step=1)  # all no-ops, none may raise
    assert runtime.debt_s == 0.0


def test_full_budget_does_not_sleep(tmp_path):
    publish(tmp_path, fraction=1.0)
    runtime = AdaptiveRuntime(job_dir=tmp_path)
    started = time.monotonic()
    for _ in range(20):
        with runtime.compute():
            time.sleep(0.001)
    assert time.monotonic() - started < 0.5
    assert runtime.debt_s <= 0.0


def test_half_budget_roughly_doubles_wall_time(tmp_path):
    publish(tmp_path, fraction=0.5)
    runtime = AdaptiveRuntime(job_dir=tmp_path, min_sleep_s=0.01)
    work_s = 0.02
    started = time.monotonic()
    for _ in range(10):
        with runtime.compute():
            time.sleep(work_s)
    elapsed = time.monotonic() - started
    assert elapsed == pytest.approx(10 * work_s * 2, rel=0.5)


def test_debt_accrues_for_the_whole_busy_stretch(tmp_path):
    """Work done outside the region still competes for the machine."""
    publish(tmp_path, fraction=0.5)
    runtime = AdaptiveRuntime(job_dir=tmp_path, min_sleep_s=10.0)  # never actually sleeps
    with runtime.compute():
        time.sleep(0.02)
    time.sleep(0.02)  # "other" loop work, outside any region
    runtime.yield_if_needed()
    # at fraction 0.5 the debt owed is ~1x the busy time, which includes both
    assert runtime.debt_s == pytest.approx(0.04, rel=0.6)


def test_stale_budget_means_full_speed(tmp_path):
    publish(tmp_path, fraction=0.1, age_s=BUDGET_STALE_AFTER_S + 10)
    runtime = AdaptiveRuntime(job_dir=tmp_path)
    started = time.monotonic()
    for _ in range(5):
        with runtime.compute():
            time.sleep(0.005)
    assert time.monotonic() - started < 0.3  # did not obey the dead controller
    assert runtime.debt_s == 0.0


def test_pause_blocks_then_releases(tmp_path):
    publish(tmp_path, should_pause=True)
    runtime = AdaptiveRuntime(job_dir=tmp_path)

    import threading
    released = threading.Event()

    def worker():
        runtime.wait_while_paused()
        released.set()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    assert not released.wait(timeout=0.5)  # still paused
    publish(tmp_path, should_pause=False)
    assert released.wait(timeout=3.0)


def test_pause_gives_up_if_the_controller_dies(tmp_path):
    """A workload must never stay paused forever because nobody is left to unpause it."""
    publish(tmp_path, should_pause=True, age_s=BUDGET_STALE_AFTER_S + 10)
    runtime = AdaptiveRuntime(job_dir=tmp_path)
    started = time.monotonic()
    runtime.wait_while_paused()
    assert time.monotonic() - started < 1.0


def test_report_writes_metrics(tmp_path):
    publish(tmp_path)
    runtime = AdaptiveRuntime(job_dir=tmp_path)
    with runtime.compute():
        pass
    runtime.report(step=3, loss=1.5, tokens=128)
    runtime._metrics.flush()
    records = MetricsTailer(tmp_path).read_new()
    assert records[0]["step"] == 3
    assert records[0]["loss"] == 1.5
    assert "ts" in records[0]
    assert "step_time_ms" in records[0]  # derived from the region timing


def test_report_never_raises(tmp_path):
    publish(tmp_path)
    runtime = AdaptiveRuntime(job_dir=tmp_path)
    runtime.report(bad=object())  # unserialisable: swallowed, not raised


def test_recommended_batch_scale_is_advisory(tmp_path):
    publish(tmp_path, memory_pressure="critical")
    assert AdaptiveRuntime(job_dir=tmp_path).recommended_batch_scale() == 0.5
    publish(tmp_path, memory_pressure="normal")
    assert AdaptiveRuntime(job_dir=tmp_path).recommended_batch_scale() == 1.0


def test_heartbeat_written_on_first_region(tmp_path):
    publish(tmp_path)
    runtime = AdaptiveRuntime(job_dir=tmp_path)
    with runtime.compute():
        pass
    assert cooperative_is_active(tmp_path)
