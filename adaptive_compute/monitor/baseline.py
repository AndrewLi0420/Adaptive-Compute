import json
import logging
import platform
import socket
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from adaptive_compute.monitor.probe import ProbeStats

log = logging.getLogger(__name__)

BASELINE_PATH = Path.home() / ".adaptive_compute" / "baseline.json"

# Used when no baseline has been recorded. Idle round-trip p95 measured on an
# Apple M3 was 0.14-0.19 ms; this fallback is deliberately several times that,
# so a missing baseline under-reports degradation rather than inventing it.
# A machine-specific baseline from `adaptive-compute baseline` is always better.
DEFAULT_P95_MS = 0.5


@dataclass(frozen=True, slots=True)
class Baseline:
    p50_ms: float
    p95_ms: float
    p99_ms: float
    wake_p95_ms: float
    sample_count: int
    recorded_at: float
    hostname: str
    python_version: str

    @classmethod
    def from_stats(cls, stats: ProbeStats) -> "Baseline":
        return cls(
            p50_ms=stats.p50_ms,
            p95_ms=stats.p95_ms,
            p99_ms=stats.p99_ms,
            wake_p95_ms=stats.wake_p95_ms,
            sample_count=stats.sample_count,
            recorded_at=time.time(),
            hostname=socket.gethostname(),
            python_version=platform.python_version(),
        )


def save_baseline(baseline: Baseline, path: Path = BASELINE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(asdict(baseline), indent=2))
    tmp.rename(path)


def load_baseline(path: Path = BASELINE_PATH) -> Baseline | None:
    """Load the recorded baseline, or None if there isn't a usable one."""
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError):
        log.warning("baseline file %s is unreadable; ignoring it", path, exc_info=True)
        return None
    try:
        baseline = Baseline(**data)
    except TypeError:
        log.warning("baseline file %s has unexpected fields; ignoring it", path)
        return None
    # The round-trip includes interpreter wakeup cost in the partner process,
    # so a baseline recorded under a different Python is not comparable.
    if baseline.python_version != platform.python_version():
        log.warning(
            "baseline was recorded on Python %s but this is %s; re-run "
            "`adaptive-compute baseline`",
            baseline.python_version,
            platform.python_version(),
        )
    return baseline
