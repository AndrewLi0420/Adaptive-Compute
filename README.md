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

## Platform

v1 targets macOS on Apple Silicon (PyTorch MPS / CPU). Platform-specific code is isolated so Linux/NVIDIA support can be added later. Single machine only.

## Status

Early development. Current milestone: **M1 — System Monitoring** (`adaptive-compute monitor`).

| Milestone | Status |
|---|---|
| 1. System monitoring | in progress |
| 2. Responsiveness probe | – |
| 3. Job runner | – |
| 4. Pressure model | – |
| 5. Basic scheduler policies | – |
| 6. Cooperative SDK | – |
| 7. Adaptive controller | – |
| 8. LLM fine-tuning demo (LoRA) | – |
| 9. Dashboard | – |
| 10. Benchmarking | – |
