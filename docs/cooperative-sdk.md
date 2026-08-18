# The cooperative SDK

```python
from adaptive_compute import adaptive

for batch in dataloader:
    with adaptive.compute():
        loss = train_step(batch)
    adaptive.report(step=step, loss=loss.item(), tokens=n)
```

Optional by construction: outside a managed job every call is a no-op, so an instrumented script runs
normally on its own.

## Why it exists

Generic throttling suspends the process with SIGSTOP wherever it happens to be. That is crude in three
specific ways, all of which cooperation removes:

| | generic (SIGSTOP) | cooperative |
|---|---|---|
| where it pauses | anywhere, including mid-GPU-command | between steps, chosen by the workload |
| controller dies mid-throttle | job frozen until `resume` | job notices staleness, continues |
| workload metrics | none available | loss, tokens, step time |
| resolution | one 50 ms tick | one training step |

The metrics matter more than they look. macOS exposes no per-process GPU usage at all, so a training
loop reporting its own step times is the only real window we have into GPU contention.

## The protocol

Files, not sockets. The controller passes a job directory in `ADAPTIVE_COMPUTE_JOB_DIR`; inside it:

- `budget.json` — controller writes, workload reads. Replaced atomically (tmp + rename), so a reader
  sees the old or the new file, never a partial one.
- `metrics.jsonl` — workload appends, controller tails from a byte offset, parsing only complete lines.
- `sdk.json` — workload heartbeat. Its presence and freshness is how the controller knows to stop
  suspending the process and let the workload pace itself.

A Unix socket would add framing and reconnection logic for no benefit at one write per second, and
files are debuggable with `cat`. This also keeps v1 free of networking abstractions.

**Staleness is the safety property.** If the controller dies, `budget.json` stops being refreshed;
after 30 seconds the workload logs once and runs at full speed. A paused workload applies the same
rule, so it can never stay paused forever because nobody is left to release it. If the workload's
heartbeat goes stale (10 s), the controller falls back to generic throttling.

## The yield arithmetic

For a region taking `c` seconds, a duty cycle of `f` implies sleeping `c × (1/f − 1)`. Applied
literally that formula fails, so the implementation adds:

- **Debt accounting** — owed time accumulates and is paid only once it exceeds a minimum sleep, so
  thousands of tiny regions produce occasional real sleeps instead of sub-millisecond ones the OS
  cannot honour.
- **Charging the whole busy stretch, not just the region** — everything the loop does between regions
  (fetching batches, reporting, the SDK itself) still competes for the machine. Measuring only the
  region undercharged badly: a 0.50 budget achieved 0.63 duty until this was fixed.
- **Subtracting actual sleep, not requested** — timer overshoot is real (~5 ms here), so the next
  cycle compensates rather than accumulating a systematic error.
- **Capped sleeps** (2 s), so one long region cannot stall the loop; the remainder stays in the debt.
- **Bounded credit**, so oversleeping cannot bank unlimited future compute.

### Minimum sleep is a measured tradeoff, not a taste

Chopping execution into millisecond slivers hits its duty target while destroying throughput, because
the SoC stays clocked down and caches stay cold. Measured at a 0.25 budget on an M3:

| min sleep | achieved duty | throughput vs unrestricted |
|---|---|---|
| 5 ms | 0.250 | 0.057 |
| 20 ms | 0.248 | 0.121 |
| 50 ms | 0.249 | 0.138 |
| 150 ms | 0.249 | 0.147 |

Every row hits the duty cycle; they differ by 2.6x in useful work. The default is 50 ms — near the top
of that curve, and with no measurable cost at a 0.5 budget (where 150 ms was actually worse). Fewer,
longer yields beat many short ones.

**The honest consequence:** at a 0.25 budget the workload gets ~0.14 of unrestricted throughput, not
0.25. Intermittent execution is less efficient than continuous execution, and this cost is real for
both throttling modes. Do not quote compute fraction as if it were a throughput fraction.

## Accuracy

Same synthetic workload, 15 s per run, on an M3:

| requested | achieved duty |
|---|---|
| 1.00 | 0.997 |
| 0.50 | 0.498 |
| 0.25 | 0.249 |

## What this does not fix

**Memory.** A cooperatively yielding workload still holds every byte it allocated; yielding between
batches frees nothing. `adaptive.recommended_batch_scale()` returns a hint (1.0 / 0.75 / 0.5 as kernel
memory pressure worsens) that the workload may choose to act on, but Adaptive Compute deliberately
never resizes batches itself — batch size changes training semantics, and doing that behind the
workload's back would be unsafe and algorithm-specific.

**Threads.** An `AdaptiveRuntime` is intended for one training loop and is not thread-safe.
