"""Turn raw telemetry into normalized, explainable resource pressure.

Design notes worth knowing before changing anything here:

* Components are kept separate and combined with max(), not a weighted sum.
  Resources are not substitutable — a machine with saturated memory is unusable
  no matter how idle the CPU is — and max() keeps every decision attributable to
  one named component. Weighted sums hide the cause and invent weights.
* CPU pressure excludes our own managed job, because a machine that is busy only
  with our background training is not contended. Memory pressure does *not*
  exclude it: memory is a fixed resource, so bytes we hold are bytes the user's
  apps cannot have.
* Missing telemetry contributes 0.0 and says so in the reasons. We never invent
  pressure from data we do not have.
* Thresholds live in PressureConfig; no magic numbers inline.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from adaptive_compute.monitor.baseline import DEFAULT_P95_MS, Baseline
from adaptive_compute.monitor.state import SystemState

log = logging.getLogger(__name__)

GB = 1024**3
MB = 1024**2


class Mode(str, Enum):
    IDLE = "IDLE"
    BACKGROUND = "BACKGROUND"
    INTERACTIVE = "INTERACTIVE"
    HIGH_PRESSURE = "HIGH_PRESSURE"
    CRITICAL = "CRITICAL"

    @property
    def severity(self) -> int:
        return _MODE_ORDER.index(self)


_MODE_ORDER = [
    Mode.IDLE,
    Mode.BACKGROUND,
    Mode.INTERACTIVE,
    Mode.HIGH_PRESSURE,
    Mode.CRITICAL,
]


@dataclass(frozen=True)
class PressureConfig:
    # -- interactive -------------------------------------------------------
    # Presence means "be careful", not "emergency": measurements in M2 showed
    # even a fully saturated CPU barely moves interactive responsiveness on
    # this hardware, so user presence alone must not crush the compute budget.
    interactive_max: float = 0.5
    interactive_active_s: float = 5.0  # idle below this => actively using
    interactive_idle_s: float = 120.0  # idle above this => user is gone

    # -- cpu ---------------------------------------------------------------
    cpu_low_percent: float = 25.0  # below this, no contention
    cpu_high_percent: float = 90.0  # at/above this, fully contended

    # -- memory ------------------------------------------------------------
    mem_avail_high_bytes: int = 4 * GB  # at/above this, no memory pressure
    mem_avail_low_bytes: int = 512 * MB  # at/below this, fully pressured
    # Kernel level floors. `warn` is deliberately mild: on a 16 GB machine warn
    # is close to a steady state, and a hard floor there would pin the budget
    # low forever. `critical` is the kernel telling us jetsam is near.
    mem_level_floor: dict[str, float] = field(
        default_factory=lambda: {"normal": 0.0, "warn": 0.35, "critical": 1.0}
    )
    # Rapid swap growth means active thrashing, which is what actually hurts;
    # a large but stable swap file does not.
    swap_growth_full_bps: float = 50 * MB  # bytes/sec that counts as full term
    swap_growth_weight: float = 0.3
    swap_history_s: float = 10.0

    # -- thermal -----------------------------------------------------------
    thermal_levels: dict[str, float] = field(
        default_factory=lambda: {
            "nominal": 0.0,
            "fair": 0.35,
            "serious": 0.8,
            "critical": 1.0,
        }
    )

    # -- responsiveness ----------------------------------------------------
    # Anchors, both empirical rather than invented: the owner confirmed the
    # machine felt fine at ~5 ms p95 (M2 validation), and ~50 ms is the widely
    # used threshold where interaction feels laggy. PROVISIONAL: we have no
    # measured "feels bad" point yet, so this component is a safety net rather
    # than a primary trigger. Recalibrate in M8 against real MPS training.
    resp_ok_ms: float = 5.0
    resp_bad_ms: float = 50.0

    # -- smoothing ---------------------------------------------------------
    # Asymmetric on purpose: back off quickly, recover cautiously. Attack is
    # tuned so a single noisy sample cannot cross a mode threshold on its own,
    # but two consecutive samples can.
    attack_alpha: float = 0.35
    release_alpha: float = 0.10

    # -- modes -------------------------------------------------------------
    mode_enter: dict[Mode, float] = field(
        default_factory=lambda: {
            Mode.BACKGROUND: 0.15,
            Mode.INTERACTIVE: 0.40,
            Mode.HIGH_PRESSURE: 0.75,
        }
    )
    mode_exit_margin: float = 0.05  # hysteresis band below the enter threshold
    min_dwell_s: float = 5.0  # minimum time before de-escalating
    escalate_samples: int = 2  # consecutive samples before raising the mode
    critical_resp_samples: int = 3  # consecutive samples before resp => CRITICAL
    critical_resp_level: float = 0.9

    reason_threshold: float = 0.3  # components at/above this get explained


@dataclass(frozen=True)
class PressureState:
    timestamp: float
    cpu: float
    memory: float
    thermal: float
    interactive: float
    responsiveness: float
    overall: float
    mode: Mode
    reasons: list[str]


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _ramp(value: float, zero_at: float, one_at: float) -> float:
    """Linear 0..1 ramp; handles either direction."""
    if zero_at == one_at:
        return 0.0 if value <= zero_at else 1.0
    return _clamp((value - zero_at) / (one_at - zero_at))


# -- individual components -------------------------------------------------
# Each returns (pressure, human explanation). Pure functions of one sample so
# they can be tested without a tracker.


def interactive_pressure(state: SystemState, cfg: PressureConfig) -> tuple[float, str]:
    idle = state.user_idle_seconds
    if idle is None:
        return 0.0, "user activity unknown"
    if idle <= cfg.interactive_active_s:
        return cfg.interactive_max, f"user active ({idle:.0f}s since input)"
    decay = 1.0 - _ramp(idle, cfg.interactive_active_s, cfg.interactive_idle_s)
    value = cfg.interactive_max * decay
    if value <= 0:
        return 0.0, f"user idle for {idle:.0f}s"
    return value, f"recent user activity ({idle:.0f}s since input)"


def cpu_pressure(state: SystemState, cfg: PressureConfig) -> tuple[float, str]:
    if state.cpu_utilization is None:
        return 0.0, "cpu telemetry unavailable"
    # Scale mismatch to watch out for: cpu_utilization is machine-wide 0-100,
    # while psutil reports process CPU as a share of one core (800% = 8 cores).
    # Convert the process figure to the machine-wide scale before subtracting.
    ncpu = state.cpu_count_logical or 1
    process_share = (state.process_cpu_percent or 0.0) / ncpu
    # Exclude our own managed job: a machine busy only with our background
    # training is not contended, and must not throttle itself into a spiral.
    others_percent = _clamp(state.cpu_utilization - process_share, 0.0, 100.0)
    value = _ramp(others_percent, cfg.cpu_low_percent, cfg.cpu_high_percent)
    return value, f"other processes using {others_percent:.0f}% cpu"


def memory_pressure(
    state: SystemState, swap_growth_bps: float | None, cfg: PressureConfig
) -> tuple[float, str]:
    if state.memory_available_bytes is None and state.memory_pressure is None:
        return 0.0, "memory telemetry unavailable"

    floor = 0.0
    level_note = ""
    if state.memory_pressure is not None:
        floor = cfg.mem_level_floor.get(state.memory_pressure, 0.0)
        level_note = f"kernel memory pressure: {state.memory_pressure}"

    avail_term = 0.0
    avail_note = ""
    if state.memory_available_bytes is not None:
        avail_term = 1.0 - _ramp(
            state.memory_available_bytes, cfg.mem_avail_low_bytes, cfg.mem_avail_high_bytes
        )
        avail_note = f"{state.memory_available_bytes / GB:.1f} GB available"

    growth_term = 0.0
    growth_note = ""
    if swap_growth_bps is not None and swap_growth_bps > 0:
        growth_term = (
            _clamp(swap_growth_bps / cfg.swap_growth_full_bps) * cfg.swap_growth_weight
        )
        if growth_term > 0.01:
            growth_note = f"swap growing {swap_growth_bps / MB:.0f} MB/s"

    value = _clamp(max(floor, avail_term) + growth_term)
    note = ", ".join(n for n in (level_note, avail_note, growth_note) if n)
    return value, note or "memory nominal"


def thermal_pressure(state: SystemState, cfg: PressureConfig) -> tuple[float, str]:
    if state.thermal_state is None:
        return 0.0, "thermal state unavailable"
    value = cfg.thermal_levels.get(state.thermal_state, 0.0)
    return value, f"thermal state: {state.thermal_state}"


def responsiveness_pressure(
    state: SystemState, baseline: Baseline | None, cfg: PressureConfig
) -> tuple[float, str]:
    p95 = state.responsiveness_latency_ms
    if p95 is None:
        return 0.0, "responsiveness unavailable"
    base = baseline.p95_ms if baseline is not None else DEFAULT_P95_MS
    # Excess over this machine's own baseline, against absolute anchors. A
    # ratio would be misleading: the baseline is ~0.2 ms, so 10x baseline is
    # still imperceptible.
    excess = max(0.0, p95 - base)
    value = _ramp(excess, cfg.resp_ok_ms, cfg.resp_bad_ms)
    return value, f"responsiveness p95 {p95:.1f} ms ({p95 / base:.0f}x baseline)"


class PressureTracker:
    """Streams SystemState -> PressureState, holding smoothing and history.

    Concurrency: single-threaded, driven from the control loop that owns the
    sampler. Time comes from sample timestamps, never from the wall clock, so
    behaviour is deterministic and testable.
    """

    def __init__(self, config: PressureConfig | None = None, baseline: Baseline | None = None):
        self.cfg = config or PressureConfig()
        self.baseline = baseline
        self._smoothed: dict[str, float] = {}
        self._swap_history: deque[tuple[float, int]] = deque()
        self._mode = Mode.IDLE
        self._mode_since: float | None = None
        self._critical_resp_streak = 0
        self._escalate_streak = 0

    @property
    def mode(self) -> Mode:
        return self._mode

    def update(self, state: SystemState) -> PressureState:
        cfg = self.cfg
        swap_rate = self._swap_growth_bps(state)

        raw: dict[str, tuple[float, str]] = {
            "cpu": cpu_pressure(state, cfg),
            "memory": memory_pressure(state, swap_rate, cfg),
            "thermal": thermal_pressure(state, cfg),
            "interactive": interactive_pressure(state, cfg),
            "responsiveness": responsiveness_pressure(state, self.baseline, cfg),
        }
        smoothed = {name: self._smooth(name, value) for name, (value, _) in raw.items()}
        overall = max(smoothed.values())

        # Evaluated exactly once per sample: it advances the responsiveness
        # streak counter, so calling it twice would double-count.
        critical = self._critical_condition(state, smoothed)
        mode = self._next_mode(state, overall, critical)
        reasons = self._reasons(raw, smoothed, critical)
        return PressureState(
            timestamp=state.timestamp,
            cpu=smoothed["cpu"],
            memory=smoothed["memory"],
            thermal=smoothed["thermal"],
            interactive=smoothed["interactive"],
            responsiveness=smoothed["responsiveness"],
            overall=overall,
            mode=mode,
            reasons=reasons,
        )

    # -- internals ---------------------------------------------------------

    def _smooth(self, name: str, value: float) -> float:
        previous = self._smoothed.get(name)
        if previous is None:
            self._smoothed[name] = value
            return value
        alpha = self.cfg.attack_alpha if value > previous else self.cfg.release_alpha
        updated = alpha * value + (1 - alpha) * previous
        self._smoothed[name] = updated
        return updated

    def _swap_growth_bps(self, state: SystemState) -> float | None:
        if state.swap_used_bytes is None:
            return None
        history = self._swap_history
        history.append((state.timestamp, state.swap_used_bytes))
        cutoff = state.timestamp - self.cfg.swap_history_s
        while len(history) > 2 and history[0][0] < cutoff:
            history.popleft()
        if len(history) < 2:
            return None
        (t0, b0), (t1, b1) = history[0], history[-1]
        if t1 <= t0:
            return None
        return (b1 - b0) / (t1 - t0)

    def _critical_condition(self, state: SystemState, smoothed: dict[str, float]) -> str | None:
        """Conditions that bypass smoothing entirely: safety is asymmetric."""
        if state.memory_pressure == "critical":
            return "kernel reports critical memory pressure"
        if state.thermal_state == "critical":
            return "thermal state critical"
        if smoothed["responsiveness"] >= self.cfg.critical_resp_level:
            self._critical_resp_streak += 1
            if self._critical_resp_streak >= self.cfg.critical_resp_samples:
                return "responsiveness severely degraded"
        else:
            self._critical_resp_streak = 0
        return None

    def _next_mode(self, state: SystemState, overall: float, critical: str | None) -> Mode:
        cfg = self.cfg
        if self._mode_since is None:
            self._mode_since = state.timestamp

        if critical is not None:
            target = Mode.CRITICAL
        else:
            target = Mode.IDLE
            for mode in (Mode.BACKGROUND, Mode.INTERACTIVE, Mode.HIGH_PRESSURE):
                if overall >= cfg.mode_enter[mode]:
                    target = mode

        if target is Mode.CRITICAL:
            # Safety conditions bypass confirmation entirely.
            self._escalate_streak = 0
            self._set_mode(target, state.timestamp)
        elif target.severity > self._mode.severity:
            # Escalation needs confirmation across consecutive samples, so a
            # single noisy sample cannot move the mode (and a stray spike does
            # not cause seconds of throttling while the EMA decays).
            self._escalate_streak += 1
            if self._escalate_streak >= cfg.escalate_samples:
                self._set_mode(target, state.timestamp)
        else:
            self._escalate_streak = 0
            if target.severity < self._mode.severity:
                since = self._mode_since if self._mode_since is not None else state.timestamp
                if state.timestamp - since >= cfg.min_dwell_s and self._may_exit(overall):
                    self._set_mode(target, state.timestamp)
        return self._mode

    def _may_exit(self, overall: float) -> bool:
        """Hysteresis: leaving a mode needs more than just dipping below entry."""
        enter = self.cfg.mode_enter.get(self._mode)
        if enter is None:  # IDLE has no floor; CRITICAL exits on condition clearing
            return True
        return overall < enter - self.cfg.mode_exit_margin

    def _set_mode(self, mode: Mode, timestamp: float) -> None:
        if mode is not self._mode:
            log.info("pressure mode %s -> %s", self._mode.value, mode.value)
            self._mode = mode
            self._mode_since = timestamp

    def _reasons(
        self,
        raw: dict[str, tuple[float, str]],
        smoothed: dict[str, float],
        critical: str | None,
    ) -> list[str]:
        reasons: list[str] = []
        if critical is not None:
            reasons.append(critical)
        # Explain by smoothed value (what actually drives decisions) but report
        # the detail from the current sample.
        for name, value in sorted(smoothed.items(), key=lambda kv: -kv[1]):
            if value >= self.cfg.reason_threshold:
                reasons.append(f"{raw[name][1]} [{name} {value:.2f}]")
        if not reasons:
            reasons.append("no significant pressure")
        return reasons
