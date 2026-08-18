"""Adaptive Compute: a resource-aware local compute runtime for ML workloads.

Training scripts only need the cooperative runtime:

    from adaptive_compute import adaptive

    for batch in dataloader:
        with adaptive.compute():
            loss = train_step(batch)
        adaptive.report(loss=loss.item(), tokens=n)

Importing this is safe outside a managed job: every call becomes a no-op.
"""

from adaptive_compute.sdk import AdaptiveRuntime, adaptive

__all__ = ["AdaptiveRuntime", "adaptive"]
