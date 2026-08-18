"""AIMD controller: glide toward the target instead of stepping to it.

The threshold policy jumps straight to a table value the moment the mode
changes. This one moves toward the same targets gradually, with deliberately
asymmetric rates borrowed from TCP congestion control:

    additive increase, multiplicative decrease

Backing off is multiplicative because contention is a problem the user is
feeling right now; recovering is additive because probing for spare capacity
should be cautious. That asymmetry is the same safety argument the pressure
model makes, applied to the actuator rather than the sensor.

**Why the targets come from the mode, not from raw pressure.** The plan for
this milestone specified bands on `overall` (increase below 0.3, hold to 0.6,
decrease above). That breaks on a machine like this one: chronic kernel memory
`warn` keeps `overall` at ~0.35 permanently, so after any backoff the
controller would sit in the hold band forever and never recover. The mode
machine already encodes the judgement that this state is the machine's normal
resting condition (BACKGROUND, target 0.85), so the controller inherits that
calibration rather than re-deriving a second, redundant set of thresholds on
the same signal.

**Why no output smoothing.** The plan also called for an EMA over the published
fraction. AIMD's update rule already bounds how far the budget can move in one
sample, so an extra filter would only delay backoff — the one direction that
must stay fast — while buying nothing.

**Why not a PID controller.** The plant here is `max()` over five heterogeneous,
partly-unavailable signals with no physical units in common; an integral term
over that has no meaningful interpretation, and a derivative term would amplify
sampling noise. AIMD is stable, explainable in one sentence, and every decision
it makes can be attributed to a named mode and rate.
"""

from dataclasses import dataclass, field

from adaptive_compute.scheduler.policy import (
    MAX_COMPUTE_FRACTION,
    MIN_COMPUTE_FRACTION,
    DEFAULT_MODE_TABLE,
    ResourceBudget,
)
from adaptive_compute.scheduler.pressure import Mode, PressureState


@dataclass(frozen=True)
class AimdConfig:
    # Additive increase per sample. Larger than the 0.05 originally planned
    # because M6 measured that low compute fractions are disproportionately
    # unproductive (a 0.25 budget yields ~0.14 of unrestricted throughput), so
    # lingering on the way back up costs more than it protects.
    increase_step: float = 0.10
    decrease_factor: float = 0.70  # multiplicative, per sample
    min_fraction: float = MIN_COMPUTE_FRACTION
    max_fraction: float = MAX_COMPUTE_FRACTION
    # After a CRITICAL pause, come back at half of what we were doing rather
    # than at full speed, so we do not immediately recreate the pressure.
    resume_factor: float = 0.5
    targets: dict[Mode, float] = field(default_factory=lambda: dict(DEFAULT_MODE_TABLE))


class AimdPolicy:
    """Stateful, unlike the baseline policies: it carries the current fraction."""

    name = "aimd"

    def __init__(self, config: AimdConfig | None = None):
        self.cfg = config or AimdConfig()
        self._fraction = self.cfg.max_fraction
        self._paused = False
        self._pre_pause_fraction = self.cfg.max_fraction

    @property
    def fraction(self) -> float:
        return self._fraction

    def decide(self, pressure: PressureState, previous: ResourceBudget) -> ResourceBudget:
        cfg = self.cfg

        if pressure.mode is Mode.CRITICAL:
            if not self._paused:
                self._pre_pause_fraction = self._fraction
                self._paused = True
            self._fraction = cfg.min_fraction
            return ResourceBudget(
                compute_fraction=cfg.min_fraction,
                should_pause=True,
                reasons=["aimd: mode CRITICAL -> pause"] + pressure.reasons,
            )

        if self._paused:
            self._paused = False
            self._fraction = self._clamp(self._pre_pause_fraction * cfg.resume_factor)

        target = self._clamp(cfg.targets.get(pressure.mode, cfg.max_fraction))
        previous_fraction = self._fraction

        if self._fraction > target:
            # multiplicative decrease, never overshooting past the target
            self._fraction = max(target, self._fraction * cfg.decrease_factor)
            movement = "backing off"
        elif self._fraction < target:
            self._fraction = min(target, self._fraction + cfg.increase_step)
            movement = "recovering"
        else:
            movement = "steady"

        self._fraction = self._clamp(self._fraction)
        reason = (f"aimd: mode {pressure.mode.value} target {target:.2f}, {movement} "
                  f"{previous_fraction:.2f} -> {self._fraction:.2f}")
        return ResourceBudget(
            compute_fraction=self._fraction,
            reasons=[reason] + pressure.reasons,
        )

    def _clamp(self, value: float) -> float:
        return max(self.cfg.min_fraction, min(self.cfg.max_fraction, value))
