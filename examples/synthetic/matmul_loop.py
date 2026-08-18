"""A synthetic training loop for validating cooperative throttling.

Stands in for a real training job: fixed-size units of work in a loop, with a
compute region and reported metrics. Uses NumPy if available and pure Python
otherwise, so it has no hard dependency on a ML stack.

    venv/bin/adaptive-compute run --policy fixed --fraction 0.5 -- \\
        venv/bin/python examples/synthetic/matmul_loop.py --seconds 30
"""

import argparse
import time

from adaptive_compute import adaptive

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised only without numpy
    np = None


def make_step(size: int):
    """Return a callable doing one fixed unit of work, and its 'token' count."""
    if np is not None:
        a = np.random.rand(size, size).astype("float32")
        b = np.random.rand(size, size).astype("float32")
        return (lambda: float(np.dot(a, b).sum())), size * size
    return (lambda: sum(i * i for i in range(size * 200))), size * 200


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=30.0, help="wall-clock duration")
    parser.add_argument("--size", type=int, default=256, help="work unit size")
    parser.add_argument("--report-every", type=int, default=100,
                        help="steps between metric reports; this loop is far faster than "
                             "real training, so reporting every step would flood telemetry")
    args = parser.parse_args()

    step, tokens_per_step = make_step(args.size)
    print(f"cooperative={adaptive.active}  numpy={np is not None}", flush=True)

    started = time.monotonic()
    cpu_started = time.process_time()
    steps = 0
    loss = 4.0

    while time.monotonic() - started < args.seconds:
        with adaptive.compute():
            step()
        steps += 1
        loss *= 0.999  # a plausible-looking curve; this is not real training
        if steps % args.report_every == 0:
            adaptive.report(step=steps, loss=loss, tokens=tokens_per_step * args.report_every)

    wall = time.monotonic() - started
    cpu = time.process_time() - cpu_started
    print(f"steps={steps} wall={wall:.1f}s cpu={cpu:.1f}s duty={cpu / wall:.3f} "
          f"steps_per_s={steps / wall:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
