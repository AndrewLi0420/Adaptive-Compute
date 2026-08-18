"""Scheduling policies: PressureState -> ResourceBudget.

Policies are pure decision functions, deliberately free of any knowledge about
processes or signals. That separation is what lets them be unit-tested against
synthetic pressure and swapped for benchmarking (M10), and it is where the
adaptive controller (M7) will slot in beside the baselines.
"""

from dataclasses import dataclass, field
from typing import Protocol

from adaptive_compute.scheduler.pressure import Mode, PressureState

MIN_COMPUTE_FRACTION = 0.10
MAX_COMPUTE_FRACTION = 1.00


@dataclass(frozen=True)
class ResourceBudget:
    """How aggressively the workload may run right now.

    compute_fraction is the share of wall time the workload may compute for.
    It is an approximation for generic processes (see process/throttle.py) and
    a directly honoured target for cooperative SDK workloads (M6).
    """

    compute_fraction: float = MAX_COMPUTE_FRACTION
    should_pause: bool = False
    cpu_worker_limit: int | None = None  # unused in v1; the SDK may honour it
    reasons: list[str] = field(default_factory=list)

    @property
    def is_throttled(self) -> bool:
        return self.should_pause or self.compute_fraction < MAX_COMPUTE_FRACTION


class Policy(Protocol):
    name: str

    def decide(self, pressure: PressureState, previous: ResourceBudget) -> ResourceBudget: ...


class UnrestrictedPolicy:
    """Baseline 1: never throttle. The control group for benchmarks."""

    name = "unrestricted"

    def decide(self, pressure: PressureState, previous: ResourceBudget) -> ResourceBudget:
        return ResourceBudget(reasons=["policy: unrestricted"])


class FixedPolicy:
    """Baseline 2: a constant fraction, ignoring the machine entirely.

    The point of comparison for the whole project: it wastes capacity when the
    machine is idle and cannot react when it is not.
    """

    name = "fixed"

    def __init__(self, fraction: float = 0.5):
        self.fraction = _clamp_fraction(fraction)

    def decide(self, pressure: PressureState, previous: ResourceBudget) -> ResourceBudget:
        return ResourceBudget(
            compute_fraction=self.fraction,
            reasons=[f"policy: fixed at {self.fraction:.2f}"],
        )


DEFAULT_MODE_TABLE: dict[Mode, float] = {
    Mode.IDLE: 1.00,
    Mode.BACKGROUND: 0.85,
    Mode.INTERACTIVE: 0.40,
    Mode.HIGH_PRESSURE: 0.15,
    Mode.CRITICAL: 0.0,  # paused; the value is unused
}


class ThresholdPolicy:
    """Baseline 3: one compute fraction per machine mode.

    Simple and completely predictable: every decision is a table lookup, so any
    surprise is a pressure-model question rather than a controller question.
    Its weakness is that it steps rather than glides — crossing a mode boundary
    changes the budget abruptly — which is the gap M7's controller addresses.
    """

    name = "threshold"

    def __init__(self, table: dict[Mode, float] | None = None):
        self.table = dict(table or DEFAULT_MODE_TABLE)

    def decide(self, pressure: PressureState, previous: ResourceBudget) -> ResourceBudget:
        if pressure.mode is Mode.CRITICAL:
            return ResourceBudget(
                compute_fraction=MIN_COMPUTE_FRACTION,
                should_pause=True,
                reasons=["policy: threshold", "mode CRITICAL -> pause"] + pressure.reasons,
            )
        fraction = _clamp_fraction(self.table.get(pressure.mode, MAX_COMPUTE_FRACTION))
        return ResourceBudget(
            compute_fraction=fraction,
            reasons=[f"policy: threshold, mode {pressure.mode.value} -> {fraction:.2f}"]
            + pressure.reasons,
        )


def _clamp_fraction(value: float) -> float:
    return max(MIN_COMPUTE_FRACTION, min(MAX_COMPUTE_FRACTION, value))


POLICIES = {
    UnrestrictedPolicy.name: UnrestrictedPolicy,
    FixedPolicy.name: FixedPolicy,
    ThresholdPolicy.name: ThresholdPolicy,
}


def build_policy(name: str, fraction: float = 0.5) -> Policy:
    if name == FixedPolicy.name:
        return FixedPolicy(fraction)
    try:
        return POLICIES[name]()
    except KeyError:
        raise ValueError(f"unknown policy {name!r}; choose from {sorted(POLICIES)}") from None
