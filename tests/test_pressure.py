import pytest

from adaptive_compute.monitor.baseline import Baseline
from adaptive_compute.monitor.state import SystemState
from adaptive_compute.scheduler.pressure import (
    GB,
    MB,
    Mode,
    PressureConfig,
    PressureTracker,
    cpu_pressure,
    interactive_pressure,
    memory_pressure,
    responsiveness_pressure,
    thermal_pressure,
)

CFG = PressureConfig()
BASELINE = Baseline(
    p50_ms=0.11, p95_ms=0.19, p99_ms=0.39, wake_p95_ms=5.06,
    sample_count=290, recorded_at=0.0, hostname="test", python_version="3.14.6",
)


def state(t: float = 0.0, **kwargs) -> SystemState:
    """An unloaded machine, overridable field by field."""
    defaults = dict(
        cpu_utilization=5.0,
        cpu_count_logical=8,
        process_cpu_percent=0.0,
        memory_available_bytes=8 * GB,
        memory_pressure="normal",
        swap_used_bytes=0,
        thermal_state="nominal",
        user_idle_seconds=600.0,
        responsiveness_latency_ms=0.2,
    )
    defaults.update(kwargs)
    return SystemState(timestamp=t, **defaults)


def drive(tracker: PressureTracker, count: int, start: float = 0.0, step: float = 1.0, **kwargs):
    """Feed `count` identical samples one second apart."""
    result = None
    for i in range(count):
        result = tracker.update(state(start + i * step, **kwargs))
    return result


# -- components ------------------------------------------------------------


def test_interactive_is_capped_not_saturated():
    """User presence must mean 'be careful', not 'emergency'."""
    value, why = interactive_pressure(state(user_idle_seconds=0.5), CFG)
    assert value == CFG.interactive_max == 0.5
    assert "user active" in why


def test_interactive_decays_to_zero_when_user_leaves():
    assert interactive_pressure(state(user_idle_seconds=60), CFG)[0] == pytest.approx(0.25, abs=0.05)
    assert interactive_pressure(state(user_idle_seconds=200), CFG)[0] == 0.0


def test_interactive_unknown_idle_is_not_invented():
    value, why = interactive_pressure(state(user_idle_seconds=None), CFG)
    assert value == 0.0
    assert "unknown" in why


def test_cpu_excludes_our_own_job():
    """A machine busy only with our training is not contended."""
    busy_with_us = state(cpu_utilization=100.0, process_cpu_percent=800.0, cpu_count_logical=8)
    assert cpu_pressure(busy_with_us, CFG)[0] == 0.0


def test_cpu_process_percent_is_rescaled_to_the_machine():
    """psutil reports 400% for 4 cores; that is 50% of an 8-core machine."""
    half_ours = state(cpu_utilization=100.0, process_cpu_percent=400.0, cpu_count_logical=8)
    # others = 100 - 50 = 50% -> partway up the 25..90 ramp
    assert cpu_pressure(half_ours, CFG)[0] == pytest.approx((50 - 25) / (90 - 25), abs=0.01)


def test_cpu_pressure_rises_with_other_processes():
    assert cpu_pressure(state(cpu_utilization=20.0), CFG)[0] == 0.0
    assert cpu_pressure(state(cpu_utilization=95.0), CFG)[0] == 1.0


def test_memory_chronic_warn_is_not_crippling():
    """This machine sits at kernel 'warn' routinely; it must not pin pressure high."""
    value, _ = memory_pressure(state(memory_pressure="warn", memory_available_bytes=3 * GB), None, CFG)
    assert 0.2 < value < 0.5


def test_memory_critical_is_total():
    value, why = memory_pressure(state(memory_pressure="critical"), None, CFG)
    assert value == 1.0
    assert "critical" in why


def test_memory_rises_as_available_falls():
    low = memory_pressure(state(memory_available_bytes=1 * GB), None, CFG)[0]
    high = memory_pressure(state(memory_available_bytes=8 * GB), None, CFG)[0]
    assert high == 0.0
    assert low > 0.8


def test_swap_growth_adds_pressure_but_swap_size_alone_does_not():
    """Active thrashing hurts; a large but stable swap file does not."""
    stable = memory_pressure(state(swap_used_bytes=9 * GB), 0.0, CFG)[0]
    growing = memory_pressure(state(swap_used_bytes=9 * GB), 60 * MB, CFG)[0]
    assert stable == 0.0
    assert growing == pytest.approx(CFG.swap_growth_weight, abs=0.01)


def test_thermal_levels():
    assert thermal_pressure(state(thermal_state="nominal"), CFG)[0] == 0.0
    assert thermal_pressure(state(thermal_state="serious"), CFG)[0] == 0.8
    assert thermal_pressure(state(thermal_state=None), CFG)[0] == 0.0


def test_responsiveness_ignores_levels_that_feel_fine():
    """Owner confirmed ~5 ms p95 feels fine; it must not register as pressure."""
    assert responsiveness_pressure(state(responsiveness_latency_ms=5.0), BASELINE, CFG)[0] == 0.0
    assert responsiveness_pressure(state(responsiveness_latency_ms=0.2), BASELINE, CFG)[0] == 0.0


def test_responsiveness_ramps_toward_perceptible_lag():
    mid = responsiveness_pressure(state(responsiveness_latency_ms=30.0), BASELINE, CFG)[0]
    bad = responsiveness_pressure(state(responsiveness_latency_ms=60.0), BASELINE, CFG)[0]
    assert 0.3 < mid < 0.7
    assert bad == 1.0


def test_responsiveness_unavailable_is_zero():
    assert responsiveness_pressure(state(responsiveness_latency_ms=None), BASELINE, CFG)[0] == 0.0


# -- overall / modes -------------------------------------------------------


def test_idle_machine_stays_idle():
    tracker = PressureTracker(baseline=BASELINE)
    result = drive(tracker, 30)
    assert result.mode is Mode.IDLE
    assert result.overall < CFG.mode_enter[Mode.BACKGROUND]
    assert result.reasons == ["no significant pressure"]


def test_overall_is_max_not_sum():
    """Several mild components must not add up into a false alarm."""
    tracker = PressureTracker(baseline=BASELINE)
    result = drive(tracker, 10, memory_pressure="warn", memory_available_bytes=3 * GB,
                   thermal_state="fair", user_idle_seconds=600)
    assert result.overall == pytest.approx(max(result.memory, result.thermal), abs=1e-9)
    assert result.overall < 0.5


def test_single_noisy_sample_does_not_change_mode():
    tracker = PressureTracker(baseline=BASELINE)
    drive(tracker, 20)  # settle on an idle machine
    spike = tracker.update(state(20, cpu_utilization=100.0, user_idle_seconds=0.1))
    assert spike.mode is Mode.IDLE


def test_sustained_interactive_use_reaches_interactive_mode():
    tracker = PressureTracker(baseline=BASELINE)
    drive(tracker, 20)
    result = drive(tracker, 5, start=20, user_idle_seconds=0.5, cpu_utilization=60.0)
    assert result.mode is Mode.INTERACTIVE
    assert any("user active" in r for r in result.reasons)


def test_recovery_requires_decay_and_dwell():
    tracker = PressureTracker(baseline=BASELINE)
    drive(tracker, 10, user_idle_seconds=0.5, cpu_utilization=70.0)
    assert tracker.mode is Mode.INTERACTIVE

    # user leaves; pressure must fall gradually, not snap back on one sample
    after_one = tracker.update(state(20, user_idle_seconds=300))
    assert after_one.mode is Mode.INTERACTIVE
    recovered = drive(tracker, 40, start=21, user_idle_seconds=300)
    assert recovered.mode is Mode.IDLE


def test_release_is_slower_than_attack():
    tracker = PressureTracker(baseline=BASELINE)
    tracker.update(state(0))
    rising = tracker.update(state(1, cpu_utilization=100.0))
    falling_from = rising.cpu
    falling = tracker.update(state(2, cpu_utilization=5.0))
    assert rising.cpu == pytest.approx(CFG.attack_alpha, abs=0.02)
    assert falling.cpu == pytest.approx(falling_from * (1 - CFG.release_alpha), abs=0.02)


def test_critical_memory_escalates_immediately():
    """Safety is asymmetric: no smoothing, no dwell, on the way up."""
    tracker = PressureTracker(baseline=BASELINE)
    drive(tracker, 20)
    result = tracker.update(state(20, memory_pressure="critical"))
    assert result.mode is Mode.CRITICAL
    assert any("critical memory" in r for r in result.reasons)


def test_critical_thermal_escalates_immediately():
    tracker = PressureTracker(baseline=BASELINE)
    drive(tracker, 5)
    assert tracker.update(state(5, thermal_state="critical")).mode is Mode.CRITICAL


def test_responsiveness_needs_sustained_degradation_for_critical():
    tracker = PressureTracker(baseline=BASELINE)
    drive(tracker, 20)
    # one terrible sample is not enough, even though it is severe
    first = tracker.update(state(20, responsiveness_latency_ms=500.0))
    assert first.mode is not Mode.CRITICAL
    last = drive(tracker, 10, start=21, responsiveness_latency_ms=500.0)
    assert last.mode is Mode.CRITICAL


def test_critical_clears_after_conditions_pass():
    tracker = PressureTracker(baseline=BASELINE)
    drive(tracker, 5, memory_pressure="critical")
    assert tracker.mode is Mode.CRITICAL
    recovered = drive(tracker, 60, start=10)
    assert recovered.mode is Mode.IDLE


def test_sustained_pressure_produces_meaningful_backoff():
    tracker = PressureTracker(baseline=BASELINE)
    result = drive(tracker, 30, cpu_utilization=95.0, user_idle_seconds=0.5,
                   memory_available_bytes=1 * GB, memory_pressure="warn")
    assert result.mode is Mode.HIGH_PRESSURE
    assert result.overall > 0.75


def test_reasons_explain_the_dominant_component():
    tracker = PressureTracker(baseline=BASELINE)
    result = drive(tracker, 20, memory_available_bytes=int(0.6 * GB), memory_pressure="warn")
    assert "memory" in result.reasons[0]
    assert "GB available" in result.reasons[0]


def test_missing_telemetry_never_invents_pressure():
    tracker = PressureTracker(baseline=BASELINE)
    blank = SystemState(timestamp=0.0)
    result = tracker.update(blank)
    assert result.overall == 0.0
    assert result.mode is Mode.IDLE


def test_thresholds_come_from_config():
    strict = PressureConfig(mode_enter={Mode.BACKGROUND: 0.01, Mode.INTERACTIVE: 0.02,
                                        Mode.HIGH_PRESSURE: 0.03})
    tracker = PressureTracker(config=strict, baseline=BASELINE)
    result = drive(tracker, 10, thermal_state="fair")
    assert result.mode is Mode.HIGH_PRESSURE
