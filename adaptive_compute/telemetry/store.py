"""Persist telemetry to SQLite so runs can be compared after the fact.

Concurrency: the control loop calls record_*() from its own thread, which only
appends to a queue. Exactly one writer thread owns the sqlite connection and is
the only code that touches it — sqlite connections are not safe to share across
threads, and keeping writes off the control loop means a slow disk can never
stall scheduling.

Schema note: samples is a wide, mostly-nullable table rather than key/value
rows. Comparing runs is the whole point of storing this, and wide columns make
those queries trivial; nullable columns are also the honest representation of
telemetry that may be unavailable.
"""

import json
import logging
import queue
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from adaptive_compute.monitor.state import SystemState
from adaptive_compute.scheduler.policy import ResourceBudget
from adaptive_compute.scheduler.pressure import PressureState

log = logging.getLogger(__name__)

DB_PATH = Path.home() / ".adaptive_compute" / "telemetry.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at   REAL NOT NULL,
    ended_at     REAL,
    policy       TEXT,
    command      TEXT,
    job_id       TEXT,
    final_state  TEXT,
    notes        TEXT
);

CREATE TABLE IF NOT EXISTS samples (
    run_id                   INTEGER NOT NULL REFERENCES runs(id),
    ts                       REAL NOT NULL,
    cpu_utilization          REAL,
    load_avg_1m              REAL,
    memory_utilization       REAL,
    memory_available_bytes   INTEGER,
    swap_used_bytes          INTEGER,
    memory_pressure          TEXT,
    process_cpu_percent      REAL,
    process_memory_bytes     INTEGER,
    thermal_state            TEXT,
    user_idle_seconds        REAL,
    gpu_utilization          REAL,
    responsiveness_p50_ms    REAL,
    responsiveness_p95_ms    REAL,
    responsiveness_p99_ms    REAL,
    pressure_cpu             REAL,
    pressure_memory          REAL,
    pressure_thermal         REAL,
    pressure_interactive     REAL,
    pressure_responsiveness  REAL,
    pressure_overall         REAL,
    mode                     TEXT,
    budget_fraction          REAL,
    paused                   INTEGER,
    job_state                TEXT,
    reasons_json             TEXT
);
CREATE INDEX IF NOT EXISTS samples_run_ts ON samples(run_id, ts);

CREATE TABLE IF NOT EXISTS training_metrics (
    run_id       INTEGER NOT NULL REFERENCES runs(id),
    ts           REAL NOT NULL,
    step         INTEGER,
    loss         REAL,
    tokens       INTEGER,
    step_time_ms REAL,
    extra_json   TEXT
);
CREATE INDEX IF NOT EXISTS training_run_ts ON training_metrics(run_id, ts);
"""

_STOP = object()


@dataclass(frozen=True)
class RunInfo:
    id: int
    started_at: float


class TelemetryStore:
    """Append-only telemetry writer. Use as a context manager."""

    def __init__(self, path: Path = DB_PATH, queue_size: int = 10_000):
        self.path = path
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._thread: threading.Thread | None = None
        self._run_id: int | None = None
        self._dropped = 0

    # -- lifecycle ---------------------------------------------------------

    def start_run(self, policy: str, command: list[str], job_id: str | None = None) -> RunInfo:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = self._connect()
        with connection:
            connection.executescript(SCHEMA)
            cursor = connection.execute(
                "INSERT INTO runs (started_at, policy, command, job_id) VALUES (?, ?, ?, ?)",
                (time.time(), policy, json.dumps(command), job_id),
            )
            self._run_id = int(cursor.lastrowid)
        connection.close()

        self._thread = threading.Thread(target=self._writer, daemon=True, name="telemetry")
        self._thread.start()
        return RunInfo(id=self._run_id, started_at=time.time())

    def finish_run(self, final_state: str | None = None) -> None:
        if self._thread is not None:
            self._queue.put(_STOP)
            self._thread.join(timeout=10)
            self._thread = None
        if self._run_id is not None:
            connection = self._connect()
            with connection:
                connection.execute(
                    "UPDATE runs SET ended_at = ?, final_state = ? WHERE id = ?",
                    (time.time(), final_state, self._run_id),
                )
            connection.close()
        if self._dropped:
            log.warning("dropped %d telemetry rows because the queue was full", self._dropped)

    def __enter__(self) -> "TelemetryStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.finish_run()

    # -- recording ---------------------------------------------------------

    def record_sample(
        self,
        state: SystemState,
        pressure: PressureState | None = None,
        budget: ResourceBudget | None = None,
        job_state: str | None = None,
    ) -> None:
        if self._run_id is None:
            return
        row = (
            self._run_id, state.timestamp,
            state.cpu_utilization, state.load_avg_1m, state.memory_utilization,
            state.memory_available_bytes, state.swap_used_bytes, state.memory_pressure,
            state.process_cpu_percent, state.process_memory_bytes, state.thermal_state,
            state.user_idle_seconds, state.gpu_utilization,
            state.responsiveness_p50_ms, state.responsiveness_latency_ms,
            state.responsiveness_p99_ms,
            pressure.cpu if pressure else None,
            pressure.memory if pressure else None,
            pressure.thermal if pressure else None,
            pressure.interactive if pressure else None,
            pressure.responsiveness if pressure else None,
            pressure.overall if pressure else None,
            pressure.mode.value if pressure else None,
            budget.compute_fraction if budget else None,
            int(budget.should_pause) if budget else None,
            job_state,
            json.dumps(budget.reasons) if budget else None,
        )
        self._submit(("sample", row))

    def record_training_metric(
        self,
        ts: float,
        step: int | None = None,
        loss: float | None = None,
        tokens: int | None = None,
        step_time_ms: float | None = None,
        extra: dict | None = None,
    ) -> None:
        if self._run_id is None:
            return
        self._submit((
            "training",
            (self._run_id, ts, step, loss, tokens, step_time_ms,
             json.dumps(extra) if extra else None),
        ))

    def _submit(self, item: tuple) -> None:
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            # Telemetry must never block or crash the control loop.
            self._dropped += 1

    # -- writer thread -----------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.execute("PRAGMA journal_mode=WAL")  # readers never block the writer
        connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    def _writer(self) -> None:
        connection = self._connect()
        try:
            while True:
                item = self._queue.get()
                if item is _STOP:
                    return
                batch = [item]
                while len(batch) < 200:  # drain whatever else is waiting
                    try:
                        nxt = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if nxt is _STOP:
                        self._flush(connection, batch)
                        return
                    batch.append(nxt)
                self._flush(connection, batch)
        except Exception:
            log.exception("telemetry writer stopped")
        finally:
            connection.close()

    def _flush(self, connection: sqlite3.Connection, batch: list[tuple]) -> None:
        samples = [row for kind, row in batch if kind == "sample"]
        training = [row for kind, row in batch if kind == "training"]
        with connection:
            if samples:
                connection.executemany(
                    "INSERT INTO samples VALUES (" + ",".join("?" * 27) + ")", samples
                )
            if training:
                connection.executemany(
                    "INSERT INTO training_metrics VALUES (?, ?, ?, ?, ?, ?, ?)", training
                )
