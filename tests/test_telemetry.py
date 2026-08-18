import json
import sqlite3
import time

from adaptive_compute.monitor.state import SystemState
from adaptive_compute.scheduler.policy import ResourceBudget
from adaptive_compute.scheduler.pressure import Mode, PressureState
from adaptive_compute.telemetry import TelemetryStore

PRESSURE = PressureState(
    timestamp=1.0, cpu=0.2, memory=0.35, thermal=0.0, interactive=0.5,
    responsiveness=0.0, overall=0.5, mode=Mode.INTERACTIVE, reasons=["user active"],
)
BUDGET = ResourceBudget(compute_fraction=0.4, should_pause=False, reasons=["mode INTERACTIVE"])


def store_at(tmp_path) -> TelemetryStore:
    return TelemetryStore(path=tmp_path / "telemetry.db")


def rows(path, sql):
    connection = sqlite3.connect(path)
    try:
        return connection.execute(sql).fetchall()
    finally:
        connection.close()


def test_run_and_samples_round_trip(tmp_path):
    store = store_at(tmp_path)
    run = store.start_run(policy="threshold", command=["python", "train.py"], job_id="job-1")
    for i in range(5):
        store.record_sample(
            SystemState(timestamp=float(i), cpu_utilization=50.0, memory_pressure="warn"),
            PRESSURE, BUDGET, job_state="THROTTLED",
        )
    store.finish_run(final_state="COMPLETED")

    saved = rows(store.path, "SELECT policy, command, job_id, final_state, ended_at FROM runs")
    assert saved[0][0] == "threshold"
    assert json.loads(saved[0][1]) == ["python", "train.py"]
    assert saved[0][2] == "job-1"
    assert saved[0][3] == "COMPLETED"
    assert saved[0][4] is not None

    samples = rows(store.path, "SELECT COUNT(*), MAX(budget_fraction), MAX(mode) FROM samples")
    assert samples[0][0] == 5
    assert samples[0][1] == 0.4
    assert samples[0][2] == "INTERACTIVE"
    assert all(r[0] == run.id for r in rows(store.path, "SELECT run_id FROM samples"))


def test_unavailable_telemetry_is_stored_as_null(tmp_path):
    store = store_at(tmp_path)
    store.start_run(policy="fixed", command=["x"])
    store.record_sample(SystemState(timestamp=1.0))
    store.finish_run()
    row = rows(store.path, "SELECT cpu_utilization, thermal_state, pressure_overall FROM samples")
    assert row[0] == (None, None, None)


def test_reasons_are_queryable(tmp_path):
    store = store_at(tmp_path)
    store.start_run(policy="threshold", command=["x"])
    store.record_sample(SystemState(timestamp=1.0), PRESSURE, BUDGET)
    store.finish_run()
    reasons = json.loads(rows(store.path, "SELECT reasons_json FROM samples")[0][0])
    assert reasons == ["mode INTERACTIVE"]


def test_training_metrics_round_trip(tmp_path):
    store = store_at(tmp_path)
    store.start_run(policy="threshold", command=["x"])
    store.record_training_metric(ts=1.0, step=7, loss=1.73, tokens=512, step_time_ms=80.0,
                                 extra={"lr": 0.0002})
    store.finish_run()
    row = rows(store.path, "SELECT step, loss, tokens, step_time_ms, extra_json FROM "
                           "training_metrics")[0]
    assert row[:4] == (7, 1.73, 512, 80.0)
    assert json.loads(row[4]) == {"lr": 0.0002}


def test_runs_can_be_compared(tmp_path):
    """The reason this exists at all: comparing policies after the fact."""
    path = tmp_path / "telemetry.db"
    for policy, fraction in (("unrestricted", 1.0), ("fixed", 0.5)):
        store = TelemetryStore(path=path)
        store.start_run(policy=policy, command=["x"])
        for i in range(3):
            store.record_sample(
                SystemState(timestamp=float(i)), PRESSURE,
                ResourceBudget(compute_fraction=fraction),
            )
        store.finish_run(final_state="COMPLETED")

    compared = rows(path, "SELECT r.policy, AVG(s.budget_fraction) FROM runs r "
                          "JOIN samples s ON s.run_id = r.id GROUP BY r.id ORDER BY r.id")
    assert compared == [("unrestricted", 1.0), ("fixed", 0.5)]


def test_recording_without_a_run_is_ignored(tmp_path):
    store = store_at(tmp_path)
    store.record_sample(SystemState(timestamp=1.0))  # must not raise
    store.finish_run()


def test_queue_overflow_drops_rather_than_blocking(tmp_path):
    """Telemetry must never stall the control loop."""
    store = TelemetryStore(path=tmp_path / "t.db", queue_size=2)
    store._run_id = 1  # pretend a run is active without starting the writer
    for i in range(50):
        store.record_sample(SystemState(timestamp=float(i)))
    assert store._dropped > 0


def test_context_manager_finishes_the_run(tmp_path):
    store = store_at(tmp_path)
    with store:
        store.start_run(policy="threshold", command=["x"])
        store.record_sample(SystemState(timestamp=time.time()))
    assert rows(store.path, "SELECT ended_at FROM runs")[0][0] is not None
