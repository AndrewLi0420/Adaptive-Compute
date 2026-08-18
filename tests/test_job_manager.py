import json
import os
import signal
import sys
import time

import psutil
import pytest

from adaptive_compute.process import JobManager, JobState, new_job

pytestmark = pytest.mark.slow


def py(source: str) -> list[str]:
    return [sys.executable, "-c", source]


def wait_until(predicate, timeout=10.0, interval=0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def test_successful_job_completes(tmp_path):
    m = JobManager(py("print('hello')"), root=tmp_path)
    m.start()
    assert m.job.state is JobState.RUNNING
    assert m.wait(timeout=10) is JobState.COMPLETED
    assert m.job.exit_code == 0
    assert m.job.term_signal is None
    assert m.job.stdout_path.read_text().strip() == "hello"
    assert m.job.elapsed_s >= 0


def test_nonzero_exit_is_failed(tmp_path):
    m = JobManager(py("import sys; sys.exit(3)"), root=tmp_path)
    m.start()
    assert m.wait(timeout=10) is JobState.FAILED
    assert m.job.exit_code == 3


def test_crash_by_signal_is_failed_not_stopped(tmp_path):
    """A signal we did not send is a crash, not a controlled stop."""
    m = JobManager(py("import os, signal; os.kill(os.getpid(), signal.SIGSEGV)"), root=tmp_path)
    m.start()
    assert m.wait(timeout=10) is JobState.FAILED
    assert m.job.term_signal == signal.SIGSEGV
    assert m.job.exit_code is None


def test_stderr_is_captured(tmp_path):
    m = JobManager(py("import sys; print('bad', file=sys.stderr)"), root=tmp_path)
    m.start()
    m.wait(timeout=10)
    assert "bad" in m.job.stderr_path.read_text()


def test_child_runs_in_its_own_process_group(tmp_path):
    m = JobManager(py("import time; time.sleep(30)"), root=tmp_path)
    m.start()
    try:
        assert m.job.pgid == m.job.pid  # setsid makes the child a group leader
        assert m.job.pgid != os.getpgid(0)  # and not our group
    finally:
        m.terminate(grace_s=2)


def test_terminate_kills_grandchildren(tmp_path):
    """Signalling the group must reach processes the child spawned."""
    source = (
        "import subprocess, sys, time\n"
        "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "print(c.pid, flush=True)\n"
        "time.sleep(60)\n"
    )
    m = JobManager(py(source), root=tmp_path)
    m.start()
    assert wait_until(lambda: m.job.stdout_path.read_text().strip().isdigit())
    grandchild = psutil.Process(int(m.job.stdout_path.read_text().strip()))

    assert m.terminate(grace_s=5) is JobState.STOPPED
    assert wait_until(lambda: not grandchild.is_running()
                      or grandchild.status() == psutil.STATUS_ZOMBIE)


def test_pause_and_resume(tmp_path):
    m = JobManager(py("import time\nwhile True: time.sleep(0.01)"), root=tmp_path)
    m.start()
    try:
        proc = psutil.Process(m.job.pid)
        m.pause()
        assert m.job.state is JobState.PAUSED
        assert wait_until(lambda: proc.status() == psutil.STATUS_STOPPED)
        assert m.poll() is JobState.PAUSED  # a stopped child has not exited

        m.resume()
        assert m.job.state is JobState.RUNNING
        assert wait_until(lambda: proc.status() != psutil.STATUS_STOPPED)
    finally:
        m.terminate(grace_s=2)


def test_terminate_wakes_a_paused_job(tmp_path):
    """A SIGSTOPped process never sees SIGTERM, so it must be continued first."""
    m = JobManager(py("import time\nwhile True: time.sleep(0.01)"), root=tmp_path)
    m.start()
    m.pause()
    assert m.terminate(grace_s=5) is JobState.STOPPED


def test_sigterm_ignoring_job_is_killed(tmp_path):
    source = (
        "import signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "print('ready', flush=True)\n"
        "while True: time.sleep(0.01)\n"
    )
    m = JobManager(py(source), root=tmp_path)
    m.start()
    assert wait_until(lambda: "ready" in m.job.stdout_path.read_text())
    assert m.terminate(grace_s=1) is JobState.STOPPED
    assert m.job.term_signal == signal.SIGKILL


def test_meta_json_is_valid_at_every_stage(tmp_path):
    m = JobManager(py("print('x')"), root=tmp_path)
    assert json.loads(m.job.meta_path.read_text())["state"] == "QUEUED"
    m.start()
    running = json.loads(m.job.meta_path.read_text())
    assert running["state"] == "RUNNING"
    assert running["pid"] == m.job.pid
    assert running["pgid"] == m.job.pgid  # recoverable after a controller crash
    m.wait(timeout=10)
    done = json.loads(m.job.meta_path.read_text())
    assert done["state"] == "COMPLETED"
    assert done["exit_code"] == 0
    assert done["ended_at"] is not None


def test_context_manager_terminates_on_exit(tmp_path):
    with JobManager(py("import time; time.sleep(60)"), root=tmp_path) as m:
        pid = m.job.pid
        proc = psutil.Process(pid)
    assert m.job.state is JobState.STOPPED
    assert wait_until(lambda: not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE)


def test_operations_after_exit_are_safe(tmp_path):
    m = JobManager(py("pass"), root=tmp_path)
    m.start()
    m.wait(timeout=10)
    # the process group is gone; none of these may raise
    m.pause()
    m.resume()
    assert m.terminate() is JobState.COMPLETED
    assert m.kill() is JobState.COMPLETED
    assert m.poll() is JobState.COMPLETED


def test_double_start_raises(tmp_path):
    m = JobManager(py("pass"), root=tmp_path)
    m.start()
    with pytest.raises(RuntimeError):
        m.start()
    m.wait(timeout=10)


def test_empty_command_rejected(tmp_path):
    with pytest.raises(ValueError):
        new_job([], root=tmp_path)


def test_jobs_started_in_the_same_second_do_not_share_a_directory(tmp_path):
    a = new_job([sys.executable, "-c", "pass"], name="same", root=tmp_path)
    b = new_job([sys.executable, "-c", "pass"], name="same", root=tmp_path)
    assert a.job_dir != b.job_dir
