import pytest

from adaptive_compute.scheduler.policy import (
    MAX_COMPUTE_FRACTION,
    MIN_COMPUTE_FRACTION,
    FixedPolicy,
    ResourceBudget,
    ThresholdPolicy,
    UnrestrictedPolicy,
    build_policy,
)
from adaptive_compute.scheduler.pressure import Mode, PressureState


def pressure(mode: Mode, overall: float = 0.5, **kwargs) -> PressureState:
    fields = dict(
        timestamp=0.0, cpu=0.0, memory=0.0, thermal=0.0, interactive=0.0,
        responsiveness=0.0, overall=overall, mode=mode, reasons=["test"],
    )
    fields.update(kwargs)
    return PressureState(**fields)


def test_unrestricted_never_throttles():
    policy = UnrestrictedPolicy()
    for mode in Mode:
        budget = policy.decide(pressure(mode, overall=1.0), ResourceBudget())
        assert budget.compute_fraction == MAX_COMPUTE_FRACTION
        assert not budget.should_pause
        assert not budget.is_throttled


def test_fixed_ignores_the_machine():
    policy = FixedPolicy(0.4)
    for mode in (Mode.IDLE, Mode.HIGH_PRESSURE):
        budget = policy.decide(pressure(mode), ResourceBudget())
        assert budget.compute_fraction == 0.4
        assert not budget.should_pause


def test_fixed_fraction_is_clamped():
    assert FixedPolicy(5.0).fraction == MAX_COMPUTE_FRACTION
    assert FixedPolicy(0.0).fraction == MIN_COMPUTE_FRACTION


def test_threshold_gives_more_compute_as_pressure_falls():
    policy = ThresholdPolicy()
    fractions = [
        policy.decide(pressure(mode), ResourceBudget()).compute_fraction
        for mode in (Mode.HIGH_PRESSURE, Mode.INTERACTIVE, Mode.BACKGROUND, Mode.IDLE)
    ]
    assert fractions == sorted(fractions)  # monotonically increasing
    assert fractions[-1] == MAX_COMPUTE_FRACTION


def test_interactive_pressure_reduces_budget():
    policy = ThresholdPolicy()
    idle = policy.decide(pressure(Mode.IDLE), ResourceBudget()).compute_fraction
    busy = policy.decide(pressure(Mode.INTERACTIVE), ResourceBudget()).compute_fraction
    assert busy < idle


def test_critical_pauses():
    budget = ThresholdPolicy().decide(pressure(Mode.CRITICAL, overall=1.0), ResourceBudget())
    assert budget.should_pause
    assert budget.is_throttled
    assert any("pause" in r for r in budget.reasons)


def test_recovery_after_pressure_clears():
    policy = ThresholdPolicy()
    paused = policy.decide(pressure(Mode.CRITICAL), ResourceBudget())
    assert paused.should_pause
    recovered = policy.decide(pressure(Mode.IDLE), paused)
    assert not recovered.should_pause
    assert recovered.compute_fraction == MAX_COMPUTE_FRACTION


def test_budget_explains_itself():
    budget = ThresholdPolicy().decide(pressure(Mode.INTERACTIVE), ResourceBudget())
    assert any("threshold" in r for r in budget.reasons)
    assert "test" in budget.reasons  # pressure reasons are carried through


def test_custom_table_is_honoured():
    policy = ThresholdPolicy({Mode.IDLE: 0.5})
    assert policy.decide(pressure(Mode.IDLE), ResourceBudget()).compute_fraction == 0.5


def test_build_policy():
    assert build_policy("unrestricted").name == "unrestricted"
    assert build_policy("fixed", 0.3).fraction == 0.3
    assert build_policy("threshold").name == "threshold"
    with pytest.raises(ValueError):
        build_policy("magic")
