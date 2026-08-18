# Adaptive Compute

A resource-aware local compute runtime for ML workloads. Run long training jobs on your laptop while continuing to use it normally.

**Core objective:** maximize useful ML training throughput subject to maintaining good interactive responsiveness.

```bash
# instead of:  python train.py
adaptive-compute run python train.py
```

Adaptive Compute continuously observes system conditions (CPU, unified memory, swap, thermals, user activity, interactive responsiveness) and dynamically decides how aggressively a background training job may run:

- Machine idle → training approaches unrestricted performance
- User working → training yields resources
- Severe pressure → aggressive throttle or safe pause

## How it works

```
System Monitor → Pressure Model → Adaptive Scheduler → Resource Budget
                                                            ↓
                              Job Manager (generic mode: any subprocess)
                              Cooperative SDK (opt-in: duty-cycled compute regions)
```

Two modes:

- **Generic mode** — wraps any subprocess; coarse control (priority, pause/resume).
- **Cooperative mode** — optional Python SDK (`from adaptive_compute import adaptive`); the training loop yields between compute regions and reports metrics (loss, tokens/sec), enabling fine-grained duty cycling.

Note on GPU: Adaptive Compute does not directly cap GPU utilization. It controls the duty cycle of cooperative training work to reduce resource contention.

## Measuring responsiveness

"Is the machine usable?" is not "is CPU below X%". Adaptive Compute measures it directly: a tiny
probe process sends a byte to a partner process and times the round trip. The partner is blocked in
`read()`, so the kernel has to wake it, schedule it, and let it reply — the same path an input event
takes to reach an app. Degradation is measured against a recorded per-machine baseline.

On the development machine (Apple M3, 8 cores) the idle p95 is 0.19 ms, and saturating all eight
cores raises it to 5.2 ms — about 32x. Costs 0.025% of the machine to run. Known limits, measured
rather than assumed: it does not detect memory-pressure sluggishness (a separate pressure signal
covers that), and it detects degradation without reliably ranking load levels.

## Platform

v1 targets macOS on Apple Silicon (PyTorch MPS / CPU). Platform-specific code is isolated so Linux/NVIDIA support can be added later. Single machine only.

## Status

Early development. Monitoring, the responsiveness probe, and the job runner are done; next up is the
pressure model. Nothing throttles anything yet — `run` supervises and measures, it does not schedule.

```bash
python3 -m venv venv
venv/bin/pip install -e ".[dev]"
venv/bin/adaptive-compute monitor            # live telemetry display
venv/bin/adaptive-compute monitor --json     # one JSON object per sample
venv/bin/adaptive-compute baseline           # record this machine's idle responsiveness
venv/bin/adaptive-compute run -- python train.py   # run a command as a managed job
venv/bin/pytest
venv/bin/python benchmarks/probe_validation.py   # show the probe responding to load
```

The venv directory is deliberately named `venv`, not `.venv`: on this dev machine a
sync agent sets the macOS `UF_HIDDEN` flag on everything inside dot-directories, and
Python ≥3.12 silently ignores hidden `.pth` files — which breaks `pip install -e`.

| Milestone | Status |
|---|---|
| 1. System monitoring | done |
| 2. Responsiveness probe | done |
| 3. Job runner | done |
| 4. Pressure model | – |
| 5. Basic scheduler policies | – |
| 6. Cooperative SDK | – |
| 7. Adaptive controller | – |
| 8. LLM fine-tuning demo (LoRA) | – |
| 9. Dashboard | – |
| 10. Benchmarking | – |
