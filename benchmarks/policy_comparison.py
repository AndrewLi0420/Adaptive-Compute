"""Compare scheduling policies under an identical, scripted interactive load.

Each policy runs the same cooperative workload through the same phases of
external CPU load, and we report what it cost and what it bought:

    throughput (steps completed)  vs  responsiveness (probe p95)

    venv/bin/python benchmarks/policy_comparison.py [--phase-seconds 20]

This is a draft of the M10 harness: it is deliberately small, and it measures
only what M7 needs to decide whether the AIMD controller earns its place beside
the threshold policy. Numbers are whatever the machine produces; nothing here
is normalised or smoothed for presentation.
"""

import argparse
import multiprocessing as mp
import re
import sqlite3
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import psutil

REPO = Path(__file__).resolve().parent.parent
DB = Path.home() / ".adaptive_compute" / "telemetry.db"


def _burn(stop: mp.Event) -> None:  # type: ignore[name-defined]
    while not stop.is_set():
        sum(i * i for i in range(50_000))


@dataclass
class Result:
    policy: str
    steps: int
    steps_per_s: float
    duty: float
    resp_p50: float
    resp_p95: float
    mean_budget: float
    budget_changes: int
    paused_fraction: float


def load_phase(workers: int, seconds: float) -> None:
    """Run `workers` CPU burners for `seconds`, then stop them."""
    stop = mp.Event()
    procs = [mp.Process(target=_burn, args=(stop,), daemon=True) for _ in range(workers)]
    for proc in procs:
        proc.start()
    time.sleep(seconds)
    stop.set()
    for proc in procs:
        proc.join(timeout=5)
        if proc.is_alive():
            proc.terminate()


def run_policy(policy: str, phase_s: float, python: str) -> Result:
    total_s = phase_s * 3
    cmd = [
        str(REPO / "venv/bin/adaptive-compute"), "run", "--quiet",
        "--policy", policy, "--name", f"bench-{policy}", "--",
        python, str(REPO / "examples/synthetic/matmul_loop.py"),
        "--seconds", str(total_s),
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(phase_s)                       # phase 1: machine idle
        load_phase(psutil.cpu_count() or 8, phase_s)  # phase 2: user is busy
        time.sleep(phase_s)                       # phase 3: idle again
        proc.wait(timeout=total_s)
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=30)

    return collect(policy)


def collect(policy: str) -> Result:
    job_dirs = sorted((Path.home() / ".adaptive_compute" / "jobs").glob(f"*bench-{policy}"))
    stdout = (job_dirs[-1] / "stdout.log").read_text()
    steps = int(re.search(r"steps=(\d+)", stdout).group(1))
    steps_per_s = float(re.search(r"steps_per_s=([\d.]+)", stdout).group(1))
    duty = float(re.search(r"duty=([\d.]+)", stdout).group(1))

    connection = sqlite3.connect(DB)
    try:
        run_id = connection.execute(
            "SELECT id FROM runs WHERE policy = ? ORDER BY id DESC LIMIT 1", (policy,)
        ).fetchone()[0]
        rows = connection.execute(
            "SELECT responsiveness_p50_ms, responsiveness_p95_ms, budget_fraction, paused "
            "FROM samples WHERE run_id = ? ORDER BY ts", (run_id,)
        ).fetchall()
    finally:
        connection.close()

    p50 = [r[0] for r in rows if r[0] is not None]
    p95 = [r[1] for r in rows if r[1] is not None]
    budgets = [r[2] for r in rows if r[2] is not None]
    paused = [r[3] for r in rows if r[3] is not None]
    changes = sum(1 for a, b in zip(budgets, budgets[1:]) if abs(b - a) > 1e-6)

    return Result(
        policy=policy,
        steps=steps,
        steps_per_s=steps_per_s,
        duty=duty,
        resp_p50=statistics.median(p50) if p50 else float("nan"),
        resp_p95=statistics.quantiles(p95, n=20)[18] if len(p95) > 20 else
        (max(p95) if p95 else float("nan")),
        mean_budget=statistics.mean(budgets) if budgets else float("nan"),
        budget_changes=changes,
        paused_fraction=(sum(paused) / len(paused)) if paused else 0.0,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-seconds", type=float, default=20.0)
    parser.add_argument("--policies", nargs="+",
                        default=["unrestricted", "fixed", "threshold", "aimd"])
    parser.add_argument("--python", default=str(REPO / "venv/bin/python"))
    args = parser.parse_args()

    print(f"phases: {args.phase_seconds:.0f}s idle / {args.phase_seconds:.0f}s cpu load / "
          f"{args.phase_seconds:.0f}s idle, per policy\n")

    results = []
    for policy in args.policies:
        print(f"running {policy} ...", flush=True)
        results.append(run_policy(policy, args.phase_seconds, args.python))
        time.sleep(3)  # let the machine settle between runs

    baseline = next((r for r in results if r.policy == "unrestricted"), None)
    header = (f"{'policy':<14}{'steps':>9}{'vs unrestr':>12}{'duty':>7}"
              f"{'resp p50':>10}{'resp p95':>10}{'budget':>8}{'changes':>9}")
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        relative = f"{r.steps / baseline.steps:.2f}x" if baseline and baseline.steps else "-"
        print(f"{r.policy:<14}{r.steps:>9}{relative:>12}{r.duty:>7.2f}"
              f"{r.resp_p50:>10.2f}{r.resp_p95:>10.2f}{r.mean_budget:>8.2f}{r.budget_changes:>9}")
    print("\nresponsiveness in ms (lower is better); changes = budget adjustments made")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
