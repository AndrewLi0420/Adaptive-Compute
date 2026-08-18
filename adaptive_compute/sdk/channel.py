"""The controller <-> workload protocol.

Files, not sockets. The controller publishes the current budget by atomically
replacing a small JSON file; the workload reads it and appends metrics to a
JSONL file. Both sides import this module so the protocol is defined once.

Why files rather than a Unix socket: an atomic rename is crash-transparent (a
reader sees the old or the new file, never a partial one), there is no
connection state to lose or re-establish if either side restarts, it costs no
protocol code, and it is debuggable with `cat`. A socket would add framing and
reconnection logic for no benefit at these rates (one write per second). It
also keeps v1 free of networking abstractions, per the spec's guidance about
the distributed future.

Staleness is the safety property that matters: if the controller dies, its
budget file stops being updated, and the workload must notice and carry on
rather than obeying a dead scheduler forever.
"""

import atexit
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

JOB_DIR_ENV = "ADAPTIVE_COMPUTE_JOB_DIR"

BUDGET_FILENAME = "budget.json"
METRICS_FILENAME = "metrics.jsonl"
HEARTBEAT_FILENAME = "sdk.json"

# How long a budget may go unrefreshed before the workload assumes the
# controller is gone. Comfortably longer than the 1 s publish interval.
BUDGET_STALE_AFTER_S = 30.0
# How long the workload's heartbeat may go unrefreshed before the controller
# stops treating it as cooperative and falls back to generic throttling.
HEARTBEAT_STALE_AFTER_S = 10.0


@dataclass(frozen=True)
class PublishedBudget:
    """What the controller tells the workload. Deliberately small and flat."""

    compute_fraction: float = 1.0
    should_pause: bool = False
    memory_pressure: str | None = None  # advisory, for recommended_batch_scale
    mode: str | None = None
    written_at: float = 0.0
    seq: int = 0

    @property
    def age_s(self) -> float:
        return max(0.0, time.time() - self.written_at)

    def is_stale(self, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        return (now - self.written_at) > BUDGET_STALE_AFTER_S


def job_dir_from_env(env: dict[str, str] | None = None) -> Path | None:
    raw = (env or os.environ).get(JOB_DIR_ENV)
    return Path(raw) if raw else None


def write_json_atomically(path: Path, payload: dict) -> None:
    """tmp + rename: a reader never observes a partially written file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)


class BudgetPublisher:
    """Controller side: publish the current budget for the workload to read."""

    def __init__(self, job_dir: Path):
        self.path = job_dir / BUDGET_FILENAME
        self._seq = 0

    def publish(
        self,
        compute_fraction: float,
        should_pause: bool,
        memory_pressure: str | None = None,
        mode: str | None = None,
    ) -> None:
        self._seq += 1
        write_json_atomically(self.path, {
            "compute_fraction": compute_fraction,
            "should_pause": should_pause,
            "memory_pressure": memory_pressure,
            "mode": mode,
            "written_at": time.time(),
            "seq": self._seq,
        })


def read_budget(path: Path) -> PublishedBudget | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    try:
        return PublishedBudget(
            compute_fraction=float(data["compute_fraction"]),
            should_pause=bool(data["should_pause"]),
            memory_pressure=data.get("memory_pressure"),
            mode=data.get("mode"),
            written_at=float(data.get("written_at", 0.0)),
            seq=int(data.get("seq", 0)),
        )
    except (KeyError, TypeError, ValueError):
        return None


class Heartbeat:
    """Workload side: advertise that a cooperative workload is alive.

    The controller uses this to decide whether to throttle by suspending the
    process (generic) or to leave it alone because it throttles itself
    (cooperative). Rewritten at most every `interval_s` so a fast training loop
    does not turn this into a write storm.
    """

    def __init__(self, job_dir: Path, interval_s: float = 2.0):
        self.path = job_dir / HEARTBEAT_FILENAME
        self.interval_s = interval_s
        self._last_write = 0.0

    def beat(self, regions: int, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_write < self.interval_s:
            return
        self._last_write = now
        try:
            write_json_atomically(self.path, {
                "pid": os.getpid(),
                "regions": regions,
                "updated_at": time.time(),
            })
        except OSError:
            pass  # never break the training loop over telemetry


def cooperative_is_active(job_dir: Path, now: float | None = None) -> bool:
    """Controller side: is a cooperative workload currently driving itself?"""
    try:
        data = json.loads((job_dir / HEARTBEAT_FILENAME).read_text())
        updated_at = float(data["updated_at"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return False
    now = time.time() if now is None else now
    return (now - updated_at) <= HEARTBEAT_STALE_AFTER_S


class MetricsWriter:
    """Workload side: append one JSON object per reported metric.

    Buffered, because a tight loop can call report() thousands of times a
    second and a file append per call is real CPU stolen from training. Flushed
    on an interval, at a record cap, and at interpreter exit; the controller
    samples once a second, so a sub-second lag costs nothing.
    """

    def __init__(self, job_dir: Path, flush_interval_s: float = 0.5, max_buffered: int = 200):
        self.path = job_dir / METRICS_FILENAME
        self.flush_interval_s = flush_interval_s
        self.max_buffered = max_buffered
        self._buffer: list[str] = []
        self._last_flush = time.monotonic()
        atexit.register(self.flush)

    def append(self, payload: dict) -> None:
        try:
            self._buffer.append(json.dumps(payload))
        except TypeError:
            return  # unserialisable metric: drop it, never raise into training
        now = time.monotonic()
        if len(self._buffer) >= self.max_buffered or now - self._last_flush >= self.flush_interval_s:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        buffered, self._buffer = self._buffer, []
        self._last_flush = time.monotonic()
        try:
            with self.path.open("a") as handle:
                handle.write("\n".join(buffered) + "\n")
        except OSError:
            pass  # reporting must never raise into the training loop


class MetricsTailer:
    """Controller side: read whatever the workload has appended since last time.

    Tracks a byte offset and only reads complete lines, so a partially flushed
    write is picked up on the next pass instead of being parsed as garbage.
    """

    def __init__(self, job_dir: Path):
        self.path = job_dir / METRICS_FILENAME
        self._offset = 0
        self._pending = ""

    def read_new(self) -> list[dict]:
        try:
            with self.path.open("r") as handle:
                handle.seek(self._offset)
                chunk = handle.read()
                self._offset = handle.tell()
        except OSError:
            return []

        self._pending += chunk
        if "\n" not in self._pending:
            return []
        *complete, self._pending = self._pending.split("\n")

        records = []
        for line in complete:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records
