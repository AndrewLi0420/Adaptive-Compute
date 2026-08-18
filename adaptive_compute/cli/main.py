import argparse
import dataclasses
import json
import logging
import signal
import sys
import time
from pathlib import Path

import psutil

from adaptive_compute.monitor import (
    Baseline,
    ResponsivenessProbe,
    Sampler,
    SystemState,
    load_baseline,
    save_baseline,
)
from adaptive_compute.monitor.baseline import BASELINE_PATH
from adaptive_compute.platform.macos import default_providers
from adaptive_compute.process import Job, JobManager, JobState
from adaptive_compute.process.manager import DEFAULT_GRACE_S
from adaptive_compute.scheduler import PressureState, PressureTracker


def _bar(pct: float, width: int = 20) -> str:
    filled = round(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)


def _fmt_bytes(n: int) -> str:
    gb = n / (1024**3)
    return f"{gb:.1f} GB" if gb >= 1 else f"{n / (1024**2):.0f} MB"


MONITOR_TITLE = "adaptive-compute monitor"


def render_pressure(pressure: PressureState) -> str:
    lines = [
        f"  MODE    {pressure.mode.value:<14} pressure {_bar(pressure.overall * 100, 10)} "
        f"{pressure.overall:.2f}",
        "  WHY",
    ]
    lines += [f"          • {reason}" for reason in pressure.reasons]
    return "\n".join(lines)


def render(state: SystemState, baseline: Baseline | None = None, title: str | None = None) -> str:
    def na(value: object, fmt: str = "{}") -> str:
        return "n/a" if value is None else fmt.format(value)

    def na_bytes(value: int | None) -> str:
        return "n/a" if value is None else _fmt_bytes(value)

    lines = [title, ""] if title else []

    if state.cpu_utilization is not None:
        lines.append(f"  CPU     {_bar(state.cpu_utilization)} {state.cpu_utilization:5.1f}%"
                     f"   load {na(state.load_avg_1m, '{:.2f}')}")
        if state.per_core_utilization:
            cores = " ".join(f"{c:3.0f}" for c in state.per_core_utilization)
            lines.append(f"  cores   [{cores}]")
    else:
        lines.append("  CPU     n/a")

    if state.memory_utilization is not None:
        lines.append(f"  Memory  {_bar(state.memory_utilization)} {state.memory_utilization:5.1f}%"
                     f"   avail {na_bytes(state.memory_available_bytes)}"
                     f"   pressure {na(state.memory_pressure)}")
    else:
        lines.append(f"  Memory  n/a   pressure {na(state.memory_pressure)}")
    lines.append(f"  Swap    {na_bytes(state.swap_used_bytes)} used")

    if state.gpu_utilization is not None:
        lines.append(f"  GPU     {_bar(state.gpu_utilization)} {state.gpu_utilization:5.1f}%   (best-effort)")
    else:
        lines.append("  GPU     n/a")

    if state.process_cpu_percent is not None:
        lines.append(f"  Process cpu {state.process_cpu_percent:.1f}%"
                     f"   mem {na_bytes(state.process_memory_bytes)}")

    lines.append("")
    lines.append(f"  Thermal    {na(state.thermal_state)}"
                 + ("   (low power mode)" if state.low_power_mode else ""))
    power = "n/a"
    if state.plugged_in is not None:
        power = ("AC" if state.plugged_in else "battery") + f"  {na(state.battery_percent, '{:.0f}%')}"
    lines.append(f"  Power      {power}")
    lines.append(f"  User idle  {na(state.user_idle_seconds, '{:.1f}s')}")

    if state.responsiveness_latency_ms is None:
        lines.append("  Response   n/a")
    else:
        line = (f"  Response   p50 {na(state.responsiveness_p50_ms, '{:.2f}')}"
                f"  p95 {state.responsiveness_latency_ms:.2f}"
                f"  p99 {na(state.responsiveness_p99_ms, '{:.2f}')} ms")
        if baseline is not None:
            line += f"   ({state.responsiveness_latency_ms / baseline.p95_ms:.1f}x baseline)"
        lines.append(line)

    lines.append("")
    lines.append(f"  sample overhead {na(state.monitor_overhead_ms, '{:.1f} ms')}")
    return "\n".join(lines)


def cmd_monitor(args: argparse.Namespace) -> int:
    providers = default_providers(pid=args.pid)
    baseline = load_baseline()
    tracker = PressureTracker(baseline=baseline)

    # --once has no sample window to build percentiles from, so the probe is
    # not started at all rather than reporting a meaningless number.
    probe = None
    if args.probe and not args.once:
        probe = ResponsivenessProbe()
        providers.append(probe)

    sampler = Sampler(providers, interval_s=args.interval)

    if args.once:
        # cpu_percent measures a delta; give it a real interval after priming
        time.sleep(min(args.interval, 1.0))
        state = sampler.sample_once()
        if args.json:
            print(json.dumps(dataclasses.asdict(state)))
        else:
            print(render(state, baseline, MONITOR_TITLE))
            print()
            print(render_pressure(tracker.update(state)))
        return 0

    def on_sample(state: SystemState) -> None:
        pressure = tracker.update(state)
        if args.json:
            print(json.dumps(dataclasses.asdict(state)), flush=True)
        else:
            # move cursor home, redraw, clear anything left below
            body = render(state, baseline, MONITOR_TITLE) + "\n\n" + render_pressure(pressure)
            sys.stdout.write("\x1b[H" + body + "\x1b[0J\n")
            sys.stdout.flush()

    if not args.json:
        sys.stdout.write("\x1b[2J\x1b[H")
    try:
        if probe is not None:
            probe.start()
        sampler.run(on_sample)
    except KeyboardInterrupt:
        pass
    finally:
        if probe is not None:
            probe.stop()
    return 0


def render_job(job: Job, state: SystemState | None, baseline: Baseline | None,
               pressure: PressureState | None = None) -> str:
    elapsed = job.elapsed_s or 0.0
    lines = [
        f"adaptive-compute run   {job.name}",
        "",
        f"  Job     {job.state.value:<10} pid {job.pid}   elapsed {elapsed:6.0f}s",
        f"  Logs    {job.job_dir}",
        "",
    ]
    if state is not None:
        lines.append(render(state, baseline))
    if pressure is not None:
        lines.append("")
        lines.append(render_pressure(pressure))
    return "\n".join(lines)



def cmd_run(args: argparse.Namespace) -> int:
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        print("run: no command given", file=sys.stderr)
        return 2

    manager = JobManager(command, name=args.name, grace_s=args.grace)
    baseline = load_baseline()
    tracker = PressureTracker(baseline=baseline)

    # Signal handling: the handler only records intent; all real work happens
    # in the control loop below, so we never terminate a child from inside a
    # signal handler. A second Ctrl-C escalates to SIGKILL.
    shutdown = {"requested": False, "hard": False}

    def on_signal(signum: int, _frame: object) -> None:
        if shutdown["requested"]:
            shutdown["hard"] = True
        shutdown["requested"] = True

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    providers = default_providers(pid=None)  # process provider added after spawn
    probe = None
    if args.probe:
        probe = ResponsivenessProbe()

    try:
        manager.start()
        # now that a pid exists, monitor the job's process tree too
        providers = default_providers(pid=manager.job.pid)
        if probe is not None:
            providers.append(probe)
            probe.start()
        sampler = Sampler(providers, interval_s=args.interval)

        if not args.quiet:
            sys.stdout.write("\x1b[2J\x1b[H")
        state: SystemState | None = None
        next_sample = time.monotonic()
        stop_deadline: float | None = None

        while True:
            # Shutdown is driven from here rather than inside terminate() so a
            # second interrupt can still escalate during the grace period.
            if shutdown["hard"]:
                print("\nsecond interrupt: killing job", file=sys.stderr)
                manager.kill()
                break
            if shutdown["requested"] and stop_deadline is None:
                print(f"\nstopping job (SIGTERM, {args.grace:.0f}s grace, "
                      "interrupt again to kill)...", file=sys.stderr)
                manager.request_terminate()
                stop_deadline = time.monotonic() + args.grace
            if manager.poll().is_terminal:
                break
            if stop_deadline is not None and time.monotonic() > stop_deadline:
                print("\ngrace period expired: killing job", file=sys.stderr)
                manager.kill()
                break

            now = time.monotonic()
            if now >= next_sample:
                next_sample = now + args.interval
                state = sampler.sample_once()
                pressure = tracker.update(state)
                if not args.quiet:
                    sys.stdout.write(
                        "\x1b[H"
                        + render_job(manager.job, state, baseline, pressure)
                        + "\x1b[0J\n"
                    )
                    sys.stdout.flush()
            # poll the child far more often than we sample, so exit is prompt
            time.sleep(0.1)
    finally:
        if probe is not None:
            probe.stop()
        if not manager.job.state.is_terminal:
            manager.terminate()

    job = manager.job
    print(f"\n{job.state.value}  exit_code={job.exit_code} signal={job.term_signal} "
          f"elapsed={job.elapsed_s:.1f}s")
    print(f"logs: {job.job_dir}")
    return 0 if job.state is JobState.COMPLETED else 1


def cmd_baseline(args: argparse.Namespace) -> int:
    """Record this machine's unloaded responsiveness for later comparison."""
    print(f"Measuring responsiveness for {args.duration:.0f}s. "
          "Leave the machine idle for a meaningful baseline.")

    cpu_samples: list[float] = []
    psutil.cpu_percent()  # prime the delta
    with ResponsivenessProbe(window_s=args.duration + 5) as probe:
        end = time.monotonic() + args.duration
        while time.monotonic() < end:
            time.sleep(1.0)
            cpu_samples.append(psutil.cpu_percent())
            if sys.stdout.isatty():
                remaining = max(0.0, end - time.monotonic())
                sys.stdout.write(f"\r  {remaining:4.0f}s remaining ")
                sys.stdout.flush()
        stats = probe.stats()
    if sys.stdout.isatty():
        print()

    if stats is None:
        print("Probe produced no usable samples; baseline not saved.", file=sys.stderr)
        return 1

    busy = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0.0
    print(f"  p50 {stats.p50_ms:.2f} ms   p95 {stats.p95_ms:.2f} ms   p99 {stats.p99_ms:.2f} ms"
          f"   ({stats.sample_count} samples)")
    print(f"  timer wakeup p95 {stats.wake_p95_ms:.2f} ms (diagnostic)")
    print(f"  mean system CPU during measurement: {busy:.1f}%")
    if busy > 25:
        print("  WARNING: the machine was busy; this baseline is not an idle baseline.")

    save_baseline(Baseline.from_stats(stats), args.output)
    print(f"Saved to {args.output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="adaptive-compute")
    sub = parser.add_subparsers(dest="command", required=True)

    monitor = sub.add_parser("monitor", help="continuously display system telemetry")
    monitor.add_argument("--interval", type=float, default=1.0, help="sampling interval in seconds")
    monitor.add_argument("--json", action="store_true", help="emit one JSON object per sample")
    monitor.add_argument("--pid", type=int, default=None, help="also track this process (and children)")
    monitor.add_argument("--once", action="store_true",
                         help="take a single sample and exit (skips the responsiveness probe)")
    monitor.add_argument("--no-probe", dest="probe", action="store_false",
                         help="do not run the responsiveness probe subprocess")
    monitor.set_defaults(func=cmd_monitor)

    run = sub.add_parser("run", help="run a command as a managed job")
    run.add_argument("--name", default=None, help="job name (default: the program name)")
    run.add_argument("--interval", type=float, default=1.0, help="telemetry sampling interval")
    run.add_argument("--grace", type=float, default=DEFAULT_GRACE_S,
                     help="seconds to wait after SIGTERM before SIGKILL")
    run.add_argument("--quiet", action="store_true", help="do not draw the live status display")
    run.add_argument("--no-probe", dest="probe", action="store_false",
                     help="do not run the responsiveness probe subprocess")
    run.add_argument("command", nargs=argparse.REMAINDER,
                     help="the command to run, e.g. -- python train.py")
    run.set_defaults(func=cmd_run)

    baseline = sub.add_parser("baseline", help="record idle responsiveness for this machine")
    baseline.add_argument("--duration", type=float, default=30.0, help="measurement seconds")
    baseline.add_argument("--output", type=Path, default=BASELINE_PATH, help="where to write it")
    baseline.set_defaults(func=cmd_baseline)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
