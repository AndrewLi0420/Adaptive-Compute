import argparse
import dataclasses
import json
import logging
import sys
import time

from adaptive_compute.monitor import Sampler, SystemState
from adaptive_compute.platform.macos import default_providers


def _bar(pct: float, width: int = 20) -> str:
    filled = round(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)


def _fmt_bytes(n: int) -> str:
    gb = n / (1024**3)
    return f"{gb:.1f} GB" if gb >= 1 else f"{n / (1024**2):.0f} MB"


def render(state: SystemState) -> str:
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
    lines.append("")
    lines.append(f"  sample overhead {na(state.monitor_overhead_ms, '{:.1f} ms')}")
    return "\n".join(lines)


def cmd_monitor(args: argparse.Namespace) -> int:
    sampler = Sampler(default_providers(pid=args.pid), interval_s=args.interval)

    if args.once:
        # cpu_percent measures a delta; give it a real interval after priming
        time.sleep(min(args.interval, 1.0))
        state = sampler.sample_once()
        if args.json:
            print(json.dumps(dataclasses.asdict(state)))
        else:
            print(render(state))
        return 0

    def on_sample(state: SystemState) -> None:
        if args.json:
            print(json.dumps(dataclasses.asdict(state)), flush=True)
        else:
            # move cursor home, redraw, clear anything left below
            sys.stdout.write("\x1b[H" + render(state) + "\x1b[0J\n")
            sys.stdout.flush()

    if not args.json:
        sys.stdout.write("\x1b[2J\x1b[H")
    try:
        sampler.run(on_sample)
    except KeyboardInterrupt:
        pass
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="adaptive-compute")
    sub = parser.add_subparsers(dest="command", required=True)

    monitor = sub.add_parser("monitor", help="continuously display system telemetry")
    monitor.add_argument("--interval", type=float, default=1.0, help="sampling interval in seconds")
    monitor.add_argument("--json", action="store_true", help="emit one JSON object per sample")
    monitor.add_argument("--pid", type=int, default=None, help="also track this process (and children)")
    monitor.add_argument("--once", action="store_true", help="take a single sample and exit")
    monitor.set_defaults(func=cmd_monitor)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
