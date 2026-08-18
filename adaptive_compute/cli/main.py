import argparse
import dataclasses
import json
import logging
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


def _bar(pct: float, width: int = 20) -> str:
    filled = round(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)


def _fmt_bytes(n: int) -> str:
    gb = n / (1024**3)
    return f"{gb:.1f} GB" if gb >= 1 else f"{n / (1024**2):.0f} MB"


def render(state: SystemState, baseline: Baseline | None = None) -> str:
    def na(value: object, fmt: str = "{}") -> str:
        return "n/a" if value is None else fmt.format(value)

    def na_bytes(value: int | None) -> str:
        return "n/a" if value is None else _fmt_bytes(value)

    lines = ["adaptive-compute monitor", ""]

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
            print(render(state, baseline))
        return 0

    def on_sample(state: SystemState) -> None:
        if args.json:
            print(json.dumps(dataclasses.asdict(state)), flush=True)
        else:
            # move cursor home, redraw, clear anything left below
            sys.stdout.write("\x1b[H" + render(state, baseline) + "\x1b[0J\n")
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

    baseline = sub.add_parser("baseline", help="record idle responsiveness for this machine")
    baseline.add_argument("--duration", type=float, default=30.0, help="measurement seconds")
    baseline.add_argument("--output", type=Path, default=BASELINE_PATH, help="where to write it")
    baseline.set_defaults(func=cmd_baseline)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
