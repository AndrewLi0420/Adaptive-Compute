import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

JOBS_ROOT = Path.home() / ".adaptive_compute" / "jobs"


class JobState(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    THROTTLED = "THROTTLED"  # set by the scheduler from M5; unused in M3
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"  # exited 0
    FAILED = "FAILED"  # exited nonzero, or died from a signal we did not send
    STOPPED = "STOPPED"  # terminated by us

    @property
    def is_terminal(self) -> bool:
        return self in (JobState.COMPLETED, JobState.FAILED, JobState.STOPPED)


@dataclass
class Job:
    """One managed workload.

    Mutable, and owned by exactly one JobManager, which is single-threaded.
    Nothing else may mutate it.
    """

    id: str
    name: str
    command: list[str]
    job_dir: Path
    state: JobState = JobState.QUEUED
    pid: int | None = None
    pgid: int | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    ended_at: float | None = None
    exit_code: int | None = None
    term_signal: int | None = None

    @property
    def stdout_path(self) -> Path:
        return self.job_dir / "stdout.log"

    @property
    def stderr_path(self) -> Path:
        return self.job_dir / "stderr.log"

    @property
    def meta_path(self) -> Path:
        return self.job_dir / "meta.json"

    @property
    def elapsed_s(self) -> float | None:
        if self.started_at is None:
            return None
        return (self.ended_at or time.time()) - self.started_at

    def to_dict(self) -> dict:
        data = asdict(self)
        data["job_dir"] = str(self.job_dir)
        data["state"] = self.state.value
        return data

    def write_meta(self) -> None:
        """Atomically publish job metadata.

        tmp + os.replace so a reader (or a crash) never sees a half-written
        file: after a controller crash this is the only record of the pgid,
        which is what lets the orphaned process group be cleaned up by hand.
        """
        tmp = self.meta_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2))
        os.replace(tmp, self.meta_path)


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", name).strip("-")
    return slug[:40] or "job"


def new_job(command: list[str], name: str | None = None, root: Path = JOBS_ROOT) -> Job:
    if not command:
        raise ValueError("command must not be empty")
    name = name or Path(command[0]).name
    base = f"{time.strftime('%Y%m%d-%H%M%S')}-{_slugify(name)}"
    root.mkdir(parents=True, exist_ok=True)
    # Second-resolution ids collide when jobs start back to back; claim the
    # directory with exist_ok=False so two jobs can never share (and overwrite)
    # one log directory.
    for suffix in range(100):
        job_id = base if suffix == 0 else f"{base}-{suffix + 1}"
        job_dir = root / job_id
        try:
            job_dir.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        return Job(id=job_id, name=name, command=list(command), job_dir=job_dir)
    raise RuntimeError(f"could not allocate a job directory under {root}")
