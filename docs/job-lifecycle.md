# Job lifecycle and process semantics

How `adaptive-compute run` manages a child process on macOS, and which guarantees it does and does
not make. Everything here is implemented in `adaptive_compute/process/` and covered by
`tests/test_job_manager.py`.

## States

```
QUEUED ──start──> RUNNING ──┬── exit 0 ────────────> COMPLETED
                            ├── exit nonzero ──────> FAILED
                            ├── killed by a signal
                            │   we did not send ───> FAILED
                            └── we terminated it ──> STOPPED
         RUNNING <──resume──> PAUSED
         THROTTLED is reserved for the scheduler (M5); the manager never sets it.
```

`exit_code` is set for normal exits and `term_signal` for signal deaths; exactly one is non-null.
The distinction between FAILED and STOPPED is *who initiated it*, not how it died: a job that
crashes with SIGSEGV is FAILED, while the same signal sent by us during shutdown is STOPPED.

## Process groups

The child is spawned with `start_new_session=True`, which calls `setsid()`: it becomes the leader of
a new session and a new process group, so `job.pgid == job.pid`.

Consequences, all deliberate:

- **Signals reach the whole tree.** `os.killpg(pgid, sig)` hits the child and everything it spawns,
  including data-loader workers. A plain `proc.terminate()` would signal only the direct child and
  leak its children.
- **Terminal signals do not reach the child behind our back.** Ctrl-C goes to the controller's
  process group only, so the controller decides what the job sees and when. Without this, Ctrl-C
  would race: the child would get SIGINT directly while the controller was still cleaning up.
- **The child has no controlling terminal.** Programs that require a TTY will see one is absent, and
  libraries that check `isatty()` (progress bars, colored output) fall back to plain output. Since
  stdout/stderr are redirected to files anyway, this costs nothing in practice.
- **Closing the terminal does not kill the job.** SIGHUP goes to the terminal's session, which the
  child is no longer part of.

## Output

`stdout` and `stderr` are redirected to `stdout.log` and `stderr.log` in the job directory, and
`stdin` is `/dev/null`. Writing to files rather than pipes means the controller never has to drain a
pipe, so a chatty job cannot deadlock by filling a pipe buffer while the controller is busy sampling.

## Shutdown

On SIGINT or SIGTERM the controller's signal handler only records intent; all real work happens in
the control loop. Nothing is killed from inside a signal handler.

1. First interrupt: SIGCONT the group if paused (see below), then SIGTERM the group, and keep the
   control loop running — telemetry continues and the job gets its grace period (`--grace`, default
   15 s) to checkpoint and exit.
2. Grace expires: SIGKILL the group.
3. Second interrupt at any point: SIGKILL immediately.

Termination is split into the non-blocking `request_terminate()` primitive and the blocking
`terminate()` convenience precisely so the escalation path stays reachable: if the controller blocked
inside `terminate()` for the full grace period, a second Ctrl-C could not be acted on.

**A paused job must be continued before it is signalled.** A SIGSTOPped process never runs, so it
never handles SIGTERM — the signal stays pending and the job appears to ignore shutdown until the
grace period expires and SIGKILL arrives. `terminate()` sends SIGCONT first.

## Orphan policy: if the controller dies, the job keeps running

If the controller is killed (SIGKILL, a crash, a panic), **the child process group continues running,
unmanaged and unthrottled.**

Why this and not the alternative:

- Training progress is expensive. Hours of work must not be destroyed because a monitoring process
  crashed. A supervisor whose failure destroys the supervised work is worse than no supervisor.
- macOS has no equivalent of Linux's `PR_SET_PDEATHSIG`, so "die with the parent" cannot be made
  reliable at the kernel level anyway. Emulating it (a watchdog polling for a live parent) would add
  a second failure mode to defend against the first.

The honest cost: after a controller crash the job runs at full speed with nothing throttling it —
exactly the situation Adaptive Compute exists to prevent — and v1 has no re-attach. This is why
`meta.json` records `pid` and `pgid` and is written atomically (tmp + `os.replace`): it is the
recovery record. Clean up an orphan by hand with

```bash
cat ~/.adaptive_compute/jobs/<job-id>/meta.json     # read pgid
kill -TERM -<pgid>                                  # negative pid = whole process group
```

For cooperative (SDK) jobs, M6 adds a stale-budget failsafe: the job notices its budget file has
stopped being updated and falls back to a safe default rather than running unmanaged forever.

## Pause and resume

`pause()`/`resume()` send SIGSTOP/SIGCONT to the process group. These are the primitives the
scheduler will use from M5.

This is a blunt instrument, and its limits should be stated plainly:

- A stopped process cannot respond to anything — not shutdown, not cleanup, not a checkpoint request.
- Stopping a process mid-GPU-command-buffer is risky, and holding memory while frozen means a paused
  job still occupies unified memory (it frees nothing).
- Nothing is coordinated with the workload's own state; it is frozen wherever it happened to be.

Cooperative pause via the SDK (M6) is the safe path for training workloads: the job yields between
compute regions, at points it chose.

## Job directory

```
~/.adaptive_compute/jobs/<YYYYmmdd-HHMMSS>-<name>/
    meta.json     job id, command, pid, pgid, state, timestamps, exit code / signal
    stdout.log
    stderr.log
```

Job ids are second-resolution, so back-to-back jobs would collide; the directory is claimed with
`mkdir(exist_ok=False)` and a `-2`, `-3` suffix is appended on collision. Two jobs never share a
directory, because that would silently overwrite the first job's logs and metadata.

## Concurrency

`JobManager` is single-threaded: every method must be called from the thread that constructed it,
which in `run` is the control loop. It spawns no threads. The `Job` dataclass is mutable and owned
solely by its manager. The only other threads in the process belong to the responsiveness probe (one
reader thread), which shares nothing with the job manager.
