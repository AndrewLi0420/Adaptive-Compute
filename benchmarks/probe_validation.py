"""Does the responsiveness probe actually respond to load?

Measures the probe under controlled conditions and prints what it saw, along
with the system state during each phase so the numbers can be read honestly.

    venv/bin/python benchmarks/probe_validation.py [--duration 20] [--memory-gb 1.0]

Milestone 2 gate: the probe must separate idle from loaded. Verify the numbers
against how the machine actually feels while this runs.
"""

import argparse
import multiprocessing as mp
import time
from dataclasses import dataclass

import psutil

from adaptive_compute.monitor import ResponsivenessProbe, Sampler
from adaptive_compute.platform.macos import default_providers


def _burn_cpu(stop: mp.Event) -> None:  # type: ignore[name-defined]
    while not stop.is_set():
        sum(i * i for i in range(10_000))


def _churn_memory(stop: mp.Event, total_bytes: int) -> None:  # type: ignore[name-defined]
    """Allocate and repeatedly touch pages so they cannot stay compressed."""
    chunk = 64 * 1024 * 1024
    blocks = [bytearray(chunk) for _ in range(max(1, total_bytes // chunk))]
    page = 16384
    while not stop.is_set():
        for block in blocks:
            for offset in range(0, len(block), page):
                block[offset] = 1
            if stop.is_set():
                break


@dataclass
class Phase:
    name: str
    cpu_workers: int
    memory_bytes: int


@dataclass
class Result:
    phase: str
    p50: float
    p95: float
    p99: float
    wake_p95: float
    samples: int
    cpu_percent: float
    mem_pressure: str | None
    swap_gb: float


def run_phase(phase: Phase, duration: float, settle: float = 3.0) -> Result:
    stop = mp.Event()
    procs: list[mp.Process] = []
    for _ in range(phase.cpu_workers):
        procs.append(mp.Process(target=_burn_cpu, args=(stop,), daemon=True))
    if phase.memory_bytes:
        procs.append(
            mp.Process(target=_churn_memory, args=(stop, phase.memory_bytes), daemon=True)
        )

    sampler = Sampler(default_providers())
    probe = ResponsivenessProbe(window_s=duration + 5)
    states = []
    try:
        for proc in procs:
            proc.start()
        probe.start()
        time.sleep(settle)  # let load ramp and the probe fill its window

        # discard the settle period so it does not dilute the phase
        probe._samples.clear()
        sampler.sample_once()  # prime cpu_percent deltas

        end = time.monotonic() + duration
        while time.monotonic() < end:
            time.sleep(1.0)
            states.append(sampler.sample_once())
        stats = probe.stats()
    finally:
        probe.stop()
        stop.set()
        for proc in procs:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()

    if stats is None:
        raise RuntimeError(f"phase {phase.name}: probe produced no usable samples")

    cpu = [s.cpu_utilization for s in states if s.cpu_utilization is not None]
    swap = [s.swap_used_bytes for s in states if s.swap_used_bytes is not None]
    return Result(
        phase=phase.name,
        p50=stats.p50_ms,
        p95=stats.p95_ms,
        p99=stats.p99_ms,
        wake_p95=stats.wake_p95_ms,
        samples=stats.sample_count,
        cpu_percent=sum(cpu) / len(cpu) if cpu else float("nan"),
        mem_pressure=states[-1].memory_pressure if states else None,
        swap_gb=(sum(swap) / len(swap) / 1024**3) if swap else float("nan"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=20.0, help="seconds per phase")
    parser.add_argument("--memory-gb", type=float, default=1.0,
                        help="memory churned in memory phases (keep modest on small machines)")
    args = parser.parse_args()

    cores = psutil.cpu_count(logical=True) or 8
    mem_bytes = int(args.memory_gb * 1024**3)
    # Idle phases are interleaved so drift in the machine's own background
    # load is visible rather than mistaken for a load response.
    phases = [
        Phase("idle 1", 0, 0),
        Phase(f"cpu x{cores // 2}", cores // 2, 0),
        Phase("idle 2", 0, 0),
        Phase(f"cpu x{cores}", cores, 0),
        Phase("idle 3", 0, 0),
        Phase(f"cpu x{cores * 2}", cores * 2, 0),
        Phase(f"memory {args.memory_gb:g}GB", 0, mem_bytes),
        Phase(f"cpu x{cores} + memory", cores, mem_bytes),
        Phase("idle 4", 0, 0),
    ]

    results = []
    for phase in phases:
        print(f"running phase: {phase.name} ...", flush=True)
        results.append(run_phase(phase, args.duration))

    header = (f"{'phase':<18}{'p50':>8}{'p95':>8}{'p99':>8}"
              f"{'wake p95':>10}{'cpu%':>8}{'swap GB':>9}  pressure")
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        print(f"{r.phase:<18}{r.p50:8.3f}{r.p95:8.3f}{r.p99:8.3f}"
              f"{r.wake_p95:10.2f}{r.cpu_percent:8.1f}{r.swap_gb:9.2f}"
              f"  {r.mem_pressure}")
    print("\nmilliseconds; p50/p95/p99 are event round-trip, wake p95 is the timer diagnostic")

    idle = [r.p95 for r in results if r.phase.startswith("idle")]
    loaded = [r.p95 for r in results if not r.phase.startswith("idle")]
    print(f"\nidle p95 spread: {min(idle):.3f} - {max(idle):.3f} ms  (measurement stability)")
    print(f"worst loaded p95: {max(loaded):.3f} ms  = {max(loaded) / (sum(idle) / len(idle)):.1f}x "
          "the mean idle p95")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
