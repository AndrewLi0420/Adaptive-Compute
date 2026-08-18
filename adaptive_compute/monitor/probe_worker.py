"""Standalone responsiveness probe worker.

Runs as its own process so that the monitor's sampling work — which holds the
GIL and can take tens of milliseconds — cannot contaminate the measurement.

Each cycle measures two things:

  rtt_ms   round-trip time to send one byte to a partner process and get it
           back. This is an *event-driven* wakeup: the partner is blocked in
           read(), the kernel must make it runnable, schedule it, and let it
           reply. It is the closest cheap analogue of "an input event arrives
           and an app handles it", and unlike a timer it is not subject to
           macOS timer coalescing. This is the headline responsiveness number.

  wake_ms  how late the OS woke us relative to a requested sleep. Classic
           scheduler wakeup latency (cyclictest-style). Kept as a corroborating
           diagnostic: on macOS it has a ~5 ms coalescing floor and a much
           smaller dynamic range than rtt.

Emits one line per cycle: "<wall_ts> <wake_ms> <rtt_ms>".

Imports nothing from adaptive_compute (stdlib only) so benchmarks can run it
directly, and so starting it stays cheap.
"""

import subprocess
import sys
import time

DEFAULT_INTERVAL_S = 0.1

# The partner process: block on one byte, echo it back. Deliberately tiny so
# its wakeup cost is dominated by scheduling, not by interpreter work.
_ECHO_SRC = (
    "import sys\n"
    "r, w = sys.stdin.buffer, sys.stdout.buffer\n"
    "while True:\n"
    "    b = r.read(1)\n"
    "    if not b:\n"
    "        break\n"
    "    w.write(b)\n"
    "    w.flush()\n"
)


def spawn_partner() -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [sys.executable, "-u", "-c", _ECHO_SRC],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )


def main(argv: list[str]) -> int:
    interval = float(argv[1]) if len(argv) > 1 else DEFAULT_INTERVAL_S
    partner = spawn_partner()
    assert partner.stdin is not None and partner.stdout is not None
    try:
        while True:
            t0 = time.monotonic()
            time.sleep(interval)
            t1 = time.monotonic()

            partner.stdin.write(b"x")
            partner.stdin.flush()
            if not partner.stdout.read(1):
                return 1  # partner died; stop reporting rather than report noise
            t2 = time.monotonic()

            try:
                sys.stdout.write(
                    f"{time.time():.3f} {(t1 - t0 - interval) * 1000:.3f} {(t2 - t1) * 1000:.3f}\n"
                )
                sys.stdout.flush()
            except (BrokenPipeError, ValueError, OSError):
                return 0  # parent went away
    finally:
        partner.kill()
        partner.wait(timeout=2)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except KeyboardInterrupt:
        sys.exit(0)
