import logging
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import IO

from adaptive_compute.process.job import JOBS_ROOT, Job, JobState, new_job

log = logging.getLogger(__name__)

DEFAULT_GRACE_S = 15.0


class JobManager:
    """Owns one child process group: lifecycle, signals, logs, metadata.

    Concurrency: single-threaded. Every method must be called from the thread
    that constructed it (in practice, the run command's control loop). It does
    not spawn threads of its own.

    Process semantics are documented in docs/job-lifecycle.md; the two that
    matter most here are that the child runs in its own session/process group
    (so signals can reach the whole tree, and terminal signals do not reach it
    behind our back), and that a controller crash deliberately leaves the child
    running.
    """

    def __init__(
        self,
        command: list[str],
        name: str | None = None,
        root: Path = JOBS_ROOT,
        grace_s: float = DEFAULT_GRACE_S,
    ):
        self.job: Job = new_job(command, name, root)
        self.grace_s = grace_s
        self._proc: subprocess.Popen[bytes] | None = None
        self._stdout: IO[bytes] | None = None
        self._stderr: IO[bytes] | None = None
        self._terminating = False  # did *we* initiate the shutdown?
        self.job.write_meta()

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._proc is not None:
            raise RuntimeError("job already started")
        self._stdout = self.job.stdout_path.open("wb")
        self._stderr = self.job.stderr_path.open("wb")
        # start_new_session => setsid(): new session and process group, so
        # os.killpg reaches the child and everything it spawns, and the
        # terminal's Ctrl-C does not reach the child directly (we forward it).
        # Cost: the child has no controlling terminal. Output is redirected to
        # files anyway, which also avoids pipe-buffer deadlocks.
        self._proc = subprocess.Popen(
            self.job.command,
            stdout=self._stdout,
            stderr=self._stderr,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        self.job.pid = self._proc.pid
        self.job.pgid = os.getpgid(self._proc.pid)
        self.job.started_at = time.time()
        self._set_state(JobState.RUNNING)
        log.info("started job %s pid=%s pgid=%s", self.job.id, self.job.pid, self.job.pgid)

    def poll(self) -> JobState:
        """Reap the child if it exited and update state. Never blocks."""
        if self._proc is None or self.job.state.is_terminal:
            return self.job.state
        returncode = self._proc.poll()  # reaps; a SIGSTOPped child is not "exited"
        if returncode is None:
            return self.job.state
        self._finalize(returncode)
        return self.job.state

    def wait(self, timeout: float | None = None) -> JobState:
        try:
            self._require_started().wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return self.job.state
        return self.poll()

    def _finalize(self, returncode: int) -> None:
        self.job.ended_at = time.time()
        if returncode < 0:
            self.job.term_signal = -returncode
            self.job.exit_code = None
        else:
            self.job.exit_code = returncode
            self.job.term_signal = None

        if self._terminating:
            state = JobState.STOPPED
        elif returncode == 0:
            state = JobState.COMPLETED
        else:
            # includes death by a signal we did not send (a crash)
            state = JobState.FAILED
        self._set_state(state)

        for stream in (self._stdout, self._stderr):
            if stream is not None:
                stream.close()
        self._stdout = self._stderr = None
        log.info("job %s -> %s (exit=%s signal=%s)", self.job.id, state.value,
                 self.job.exit_code, self.job.term_signal)

    # -- control -----------------------------------------------------------

    def pause(self) -> None:
        """SIGSTOP the process group.

        Coarse and generic-mode only. A stopped process cannot respond to
        anything, including cleanup, and stopping a process mid-GPU-command
        is risky — cooperative pause (M6) is the safe path for SDK workloads.
        """
        if self.job.state is not JobState.RUNNING:
            return
        if self._signal_group(signal.SIGSTOP):
            self._set_state(JobState.PAUSED)

    def resume(self) -> None:
        if self.job.state not in (JobState.PAUSED, JobState.THROTTLED):
            return
        if self._signal_group(signal.SIGCONT):
            self._set_state(JobState.RUNNING)

    def request_terminate(self) -> None:
        """Send SIGTERM to the group and return immediately.

        Non-blocking so the caller's control loop stays responsive during the
        grace period — that is what makes escalation (a second Ctrl-C) and
        continued telemetry possible while the job is winding down.
        """
        if self._proc is None or self.job.state.is_terminal:
            return
        self._terminating = True
        # A stopped process never sees SIGTERM, so wake it first.
        if self.job.state is JobState.PAUSED:
            self._signal_group(signal.SIGCONT)
        self._signal_group(signal.SIGTERM)

    def terminate(self, grace_s: float | None = None) -> JobState:
        """SIGTERM the group, then SIGKILL anything still alive after grace.

        Blocking convenience for callers without a control loop; anything with
        a loop should use request_terminate() and poll.
        """
        if self._proc is None or self.job.state.is_terminal:
            return self.job.state
        grace = self.grace_s if grace_s is None else grace_s
        self.request_terminate()
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            if self.poll().is_terminal:
                return self.job.state
            time.sleep(0.05)

        log.warning("job %s ignored SIGTERM for %.0fs; sending SIGKILL", self.job.id, grace)
        return self.kill()

    def kill(self) -> JobState:
        if self._proc is None or self.job.state.is_terminal:
            return self.job.state
        self._terminating = True
        self._signal_group(signal.SIGKILL)
        self._proc.wait(timeout=5)
        return self.poll()

    def _signal_group(self, sig: int) -> bool:
        """Signal the whole process group. False if it is already gone."""
        if self.job.pgid is None:
            return False
        try:
            os.killpg(self.job.pgid, sig)
            return True
        except ProcessLookupError:
            return False  # exited between our check and the signal
        except PermissionError:
            log.error("not permitted to signal process group %s", self.job.pgid)
            return False

    # -- helpers -----------------------------------------------------------

    def _set_state(self, state: JobState) -> None:
        self.job.state = state
        self.job.write_meta()

    def _require_started(self) -> subprocess.Popen[bytes]:
        if self._proc is None:
            raise RuntimeError("job not started")
        return self._proc

    def __enter__(self) -> "JobManager":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        if not self.job.state.is_terminal:
            self.terminate()
