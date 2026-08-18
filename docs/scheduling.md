# Scheduling: pressure to budget to enforcement

```
SystemState ──► PressureState ──► Policy ──► ResourceBudget ──► enforcement
 (monitor)       (pressure.py)   (policy.py)                    (throttle.py)
```

Each arrow is a separate, independently testable step. Policies know nothing about processes or
signals, which is what lets them be unit-tested against synthetic pressure and swapped out for
benchmarking.

## What `compute_fraction` means

The share of wall-clock time the workload is allowed to compute for. It is **not** a CPU quota and
**not** a GPU cap — macOS exposes neither to an unprivileged process.

> Adaptive Compute does not directly cap CPU or GPU utilization. It controls the duty cycle of the
> workload to reduce resource contention.

## Policies

| Policy | Behaviour | Why it exists |
|---|---|---|
| `unrestricted` | always 1.0 | control group; what training does today |
| `fixed` | a constant fraction | the conservative-throttle straw man: wastes capacity when idle, cannot react when busy |
| `threshold` | one fraction per mode | the first real adaptive policy: a table lookup, so every decision is predictable |

The threshold table (configurable): IDLE 1.00, BACKGROUND 0.85, INTERACTIVE 0.40, HIGH_PRESSURE 0.15,
CRITICAL pause. Its weakness is that it steps rather than glides — crossing a mode boundary changes
the budget abruptly. That gap is what M7's controller addresses.

## Enforcement for generic processes

Duty cycling by suspending the process group: over each period (default 1 s) the job runs for
`compute_fraction × period` and is SIGSTOPped for the rest. The cycler is a pure state machine
evaluated on every control-loop tick (50 ms); it never sleeps, so a shutdown request is always
serviced promptly.

Measured accuracy on an M3, using a job that spins for 20 s of wall time and reports its own CPU time:

| requested | measured | ratio |
|---|---|---|
| 1.00 | 0.976 | 0.98x |
| 0.75 | 0.734 | 0.98x |
| 0.50 | 0.483 | 0.97x |
| 0.25 | 0.235 | 0.94x |
| 0.10 | 0.090 | 0.90x |

The consistent slight undershoot is the cost of the resume itself, and it grows in relative terms as
slices shrink. Honest summary: this tracks the requested fraction within about 2–10%.

Limits worth stating plainly:

- **Resolution** is one tick, and slices shorter than 50 ms collapse to "always on" or the floor,
  because signalling a process faster than it can make progress is worse than not throttling.
- **Memory is unaffected.** A suspended process holds every byte it had. Duty cycling reduces CPU
  contention only; memory pressure has to be answered by pausing or by the workload itself.
- **SIGSTOP freezes the process wherever it is**, including mid-GPU-command-buffer. This is why
  cooperative yielding (M6) is the better path for workloads that can integrate with the SDK.

`nice` can be applied at spawn with `--nice N`, via the `/usr/bin/nice` wrapper rather than a
`preexec_fn` (forking with a preexec callback is unsafe in a process that has threads, and this one
runs the probe reader thread). Note that an unprivileged process can lower its priority but never
raise it again, so this is a one-way decision made at spawn. macOS `taskpolicy` (background QoS) is
**not** available on this machine, so it is not used.

## Hazard: a job orphaned mid-throttle stays frozen

M3 promises that killing the controller leaves the job running. Duty cycling weakens that promise: if
the controller is SIGKILLed during a stop phase, the job is left suspended indefinitely, because
nothing remains to send SIGCONT. Verified, not theorised — the child sits in state `T`.

The exposure is proportional to the stop duty: at `compute_fraction` 0.2, roughly 80% of the time.

Recovery:

```bash
adaptive-compute jobs              # shows the job's process as SUSPENDED
adaptive-compute resume <job-id>   # SIGCONT to the recorded process group
```

Ordinary shutdown paths are safe: `terminate()` always sends SIGCONT before SIGTERM, keyed on the
suspend flag rather than the job state, because a THROTTLED job is stopped for part of every period
too. A stopped process never handles SIGTERM, so skipping that step would make every throttled job
appear to ignore shutdown until the grace period expired.

## Telemetry

Every sample — system state, the five pressure components, mode, budget, reasons, job state — is
written to SQLite at `~/.adaptive_compute/telemetry.db` for later comparison between runs.

A single writer thread owns the connection and is fed by a bounded queue; the control loop only
appends. Writes therefore cannot stall scheduling, and if the queue ever fills, rows are dropped and
counted rather than blocking the loop. Telemetry is not allowed to be the reason the scheduler
stutters.
