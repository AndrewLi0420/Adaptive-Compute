"""Replay recorded telemetry through the pressure model and print the timeline.

Capture, then replay:

    venv/bin/adaptive-compute monitor --json > /tmp/telemetry.jsonl
    venv/bin/python benchmarks/pressure_replay.py /tmp/telemetry.jsonl

Reads `adaptive-compute monitor --json` output (a file, or stdin with `-`).
Deterministic and offline, so a scheduling decision can be re-examined against
the exact samples that produced it.
"""

import argparse
import json
import sys
from pathlib import Path

from adaptive_compute.monitor import load_baseline
from adaptive_compute.monitor.state import SYSTEM_STATE_FIELDS, SystemState
from adaptive_compute.scheduler import PressureTracker


def load_states(source) -> list[SystemState]:
    states = []
    for line in source:
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        states.append(SystemState(**{k: v for k, v in raw.items() if k in SYSTEM_STATE_FIELDS}))
    return states


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="JSONL file from `monitor --json`, or - for stdin")
    parser.add_argument("--reasons", action="store_true", help="print the reasons for each sample")
    args = parser.parse_args()

    source = sys.stdin if args.path == "-" else Path(args.path).open()
    states = load_states(source)
    if not states:
        print("no samples", file=sys.stderr)
        return 1

    tracker = PressureTracker(baseline=load_baseline())
    start = states[0].timestamp

    header = (f"{'t':>6}  {'cpu':>5}{'mem':>6}{'therm':>7}{'inter':>7}{'resp':>6}"
              f"{'overall':>9}  mode")
    print(header)
    print("-" * len(header))
    modes: dict[str, int] = {}
    for state in states:
        p = tracker.update(state)
        modes[p.mode.value] = modes.get(p.mode.value, 0) + 1
        print(f"{state.timestamp - start:6.0f}  {p.cpu:5.2f}{p.memory:6.2f}{p.thermal:7.2f}"
              f"{p.interactive:7.2f}{p.responsiveness:6.2f}{p.overall:9.2f}  {p.mode.value}")
        if args.reasons:
            for reason in p.reasons:
                print(f"          - {reason}")

    print(f"\n{len(states)} samples over {states[-1].timestamp - start:.0f}s")
    print("time in mode: " + ", ".join(f"{m} {c}s" for m, c in modes.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
