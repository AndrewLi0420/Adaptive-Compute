import logging
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

WORKER_PATH = Path(__file__).with_name("probe_worker.py")

MIN_SAMPLES = 20  # p95 below this is meaningless; report nothing instead
STALE_AFTER_S = 3.0  # newest sample older than this => probe is not reporting


@dataclass(frozen=True, slots=True)
class ProbeStats:
    """Percentiles over the probe's recent window.

    p50/p95/p99 are round-trip latency, the headline responsiveness number.
    wake_p95_ms is the timer-wakeup diagnostic; it is reported separately
    rather than combined because its ~5 ms macOS coalescing floor would swamp
    the round-trip signal, whose idle floor is ~0.15 ms.
    """

    p50_ms: float
    p95_ms: float
    p99_ms: float
    wake_p95_ms: float
    sample_count: int


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile.

    Nearest-rank (rather than the interpolating statistics.quantiles) keeps
    every reported number an actually-observed latency and stays defined for
    small samples.
    """
    if not values:
        raise ValueError("percentile of empty sequence")
    ordered = sorted(values)
    rank = max(1, min(len(ordered), round(pct / 100 * len(ordered))))
    return ordered[rank - 1]


class ResponsivenessProbe:
    """Measures interactive responsiveness via a subprocess probe worker.

    Latency here means round-trip time for an event-driven wakeup; see
    probe_worker for what is measured and why.

    Concurrency: the reader thread is the only writer of _samples; stats()
    snapshots it under _lock. start()/stop() are called from the owning thread.
    """

    name = "responsiveness"

    def __init__(
        self,
        interval_s: float = 0.1,
        window_s: float = 30.0,
        min_samples: int = MIN_SAMPLES,
    ):
        self.interval_s = interval_s
        self.window_s = window_s
        self.min_samples = min_samples
        # 1.5x slack so the deque never evicts samples still inside the window
        self._samples: deque[tuple[float, float, float]] = deque(
            maxlen=int(window_s / interval_s * 1.5)
        )
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[str] | None = None
        self._reader: threading.Thread | None = None
        self._died_logged = False

    def start(self) -> None:
        if self._proc is not None:
            raise RuntimeError("probe already started")
        self._proc = subprocess.Popen(
            [sys.executable, "-u", str(WORKER_PATH), str(self.interval_s)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self._reader = threading.Thread(target=self._read_loop, daemon=True, name="probe-reader")
        self._reader.start()

    def _read_loop(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            parsed = parse_sample_line(line)
            if parsed is None:
                continue
            with self._lock:
                self._samples.append(parsed)

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def stop(self) -> None:
        if self._proc is None:
            return
        self._proc.terminate()
        try:
            self._proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait(timeout=2)
        if self._reader is not None:
            self._reader.join(timeout=2)
        self._proc = None
        self._reader = None

    def stats(self, now: float | None = None) -> ProbeStats | None:
        """Percentiles over the recent window, or None if not measurable."""
        now = time.time() if now is None else now
        with self._lock:
            snapshot = list(self._samples)

        if not snapshot:
            self._log_unavailable("no samples yet")
            return None
        # max() rather than snapshot[-1]: does not assume arrival order
        if now - max(s[0] for s in snapshot) > STALE_AFTER_S:
            self._log_unavailable("probe samples are stale (worker stopped reporting?)")
            return None

        cutoff = now - self.window_s
        window = [s for s in snapshot if s[0] >= cutoff]
        if len(window) < self.min_samples:
            return None

        wake = [s[1] for s in window]
        rtt = [s[2] for s in window]
        return ProbeStats(
            p50_ms=percentile(rtt, 50),
            p95_ms=percentile(rtt, 95),
            p99_ms=percentile(rtt, 99),
            wake_p95_ms=percentile(wake, 95),
            sample_count=len(window),
        )

    def _log_unavailable(self, why: str) -> None:
        if not self.is_alive() and not self._died_logged:
            self._died_logged = True
            log.warning("responsiveness probe worker is not running: %s", why)

    def sample(self) -> dict[str, Any]:
        """Provider protocol. Absent fields stay None rather than guessed."""
        stats = self.stats()
        if stats is None:
            return {}
        return {
            "responsiveness_latency_ms": stats.p95_ms,
            "responsiveness_p50_ms": stats.p50_ms,
            "responsiveness_p99_ms": stats.p99_ms,
        }

    def __enter__(self) -> "ResponsivenessProbe":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


def parse_sample_line(line: str) -> tuple[float, float, float] | None:
    parts = line.split()
    if len(parts) != 3:
        return None
    try:
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError:
        return None
