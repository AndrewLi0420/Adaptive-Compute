import pytest

from adaptive_compute.scheduler.controller import AimdConfig, AimdPolicy
from adaptive_compute.scheduler.policy import (
    MAX_COMPUTE_FRACTION,
    MIN_COMPUTE_FRACTION,
    ResourceBudget,
)
from adaptive_compute.scheduler.pressure import Mode, PressureState


def pressure(mode: Mode, overall: float = 0.5) -> PressureState:
    return PressureState(
        timestamp=0.0, cpu=0.0, memory=0.0, thermal=0.0, interactive=0.0,
        responsiveness=0.0, overall=overall, mode=mode, reasons=["because"],
    )


def run(policy: AimdPolicy, mode: Mode, samples: int) -> list[float]:
    budget = ResourceBudget()
    fractions = []
    for _ in range(samples):
        budget = policy.decide(pressure(mode), budget)
        fractions.append(budget.compute_fraction)
    return fractions


def test_starts_unrestricted():
    assert AimdPolicy().fraction == MAX_COMPUTE_FRACTION


def test_backoff_is_multiplicative_and_fast():
    policy = AimdPolicy()
    fractions = run(policy, Mode.HIGH_PRESSURE, 6)
    assert fractions[0] == pytest.approx(0.7)  # 1.0 * 0.7 in one sample
    assert fractions[-1] == pytest.approx(0.15)  # reached the target
    assert fractions == sorted(fractions, reverse=True)  # monotone down


def test_recovery_is_additive_and_gradual():
    policy = AimdPolicy()
    run(policy, Mode.HIGH_PRESSURE, 10)  # settle at 0.15
    fractions = run(policy, Mode.IDLE, 3)
    assert fractions[0] == pytest.approx(0.25)  # +0.10, not a jump to 1.0
    assert fractions == sorted(fractions)  # monotone up


def test_recovery_eventually_reaches_full_speed():
    policy = AimdPolicy()
    run(policy, Mode.HIGH_PRESSURE, 10)
    fractions = run(policy, Mode.IDLE, 30)
    assert fractions[-1] == MAX_COMPUTE_FRACTION


def test_backoff_is_faster_than_recovery():
    """Safety is asymmetric: yield quickly, reclaim cautiously."""
    down = AimdPolicy()
    steps_down = next(i for i, f in enumerate(run(down, Mode.HIGH_PRESSURE, 50)) if f <= 0.16)

    up = AimdPolicy()
    run(up, Mode.HIGH_PRESSURE, 50)
    steps_up = next(i for i, f in enumerate(run(up, Mode.IDLE, 50)) if f >= 0.99)
    assert steps_down < steps_up


def test_never_overshoots_the_target():
    policy = AimdPolicy()
    for fraction in run(policy, Mode.INTERACTIVE, 20):
        assert fraction >= 0.40 - 1e-9
    assert policy.fraction == pytest.approx(0.40)


def test_glides_rather_than_steps():
    """The whole point versus the threshold policy: no abrupt jumps."""
    policy = AimdPolicy()
    fractions = run(policy, Mode.HIGH_PRESSURE, 8)
    jumps = [abs(b - a) for a, b in zip(fractions, fractions[1:])]
    assert max(jumps) < 0.35  # threshold would jump 1.00 -> 0.15 in one sample


def test_critical_pauses_immediately():
    policy = AimdPolicy()
    budget = policy.decide(pressure(Mode.CRITICAL), ResourceBudget())
    assert budget.should_pause
    assert any("CRITICAL" in r for r in budget.reasons)


def test_resume_after_pause_is_conservative():
    """Coming back at full speed would just recreate the pressure."""
    policy = AimdPolicy()
    run(policy, Mode.IDLE, 5)  # at 1.0
    policy.decide(pressure(Mode.CRITICAL), ResourceBudget())
    resumed = policy.decide(pressure(Mode.IDLE), ResourceBudget())
    assert resumed.compute_fraction < 1.0
    assert not resumed.should_pause
    assert resumed.compute_fraction == pytest.approx(0.5 + 0.10)  # half, then one step up


def test_alternating_pressure_does_not_oscillate_wildly():
    policy = AimdPolicy()
    budget = ResourceBudget()
    fractions = []
    for i in range(20):
        mode = Mode.IDLE if i % 2 == 0 else Mode.INTERACTIVE
        budget = policy.decide(pressure(mode), budget)
        fractions.append(budget.compute_fraction)
    swing = max(fractions[4:]) - min(fractions[4:])
    assert swing < 0.35  # settles into a narrow band rather than slamming up and down


def test_sustained_pressure_produces_meaningful_backoff():
    policy = AimdPolicy()
    assert run(policy, Mode.HIGH_PRESSURE, 15)[-1] <= 0.15


def test_stays_within_bounds():
    policy = AimdPolicy()
    for mode in Mode:
        for fraction in run(policy, mode, 10):
            assert MIN_COMPUTE_FRACTION <= fraction <= MAX_COMPUTE_FRACTION


def test_constants_come_from_config():
    policy = AimdPolicy(AimdConfig(increase_step=0.5, decrease_factor=0.1))
    assert run(policy, Mode.HIGH_PRESSURE, 1)[0] == pytest.approx(0.15)  # 1.0*0.1 clamped
    assert run(policy, Mode.IDLE, 1)[0] == pytest.approx(0.65)  # +0.5


def test_reasons_name_the_mode_and_the_movement():
    policy = AimdPolicy()
    budget = policy.decide(pressure(Mode.HIGH_PRESSURE), ResourceBudget())
    reason = budget.reasons[0]
    assert "HIGH_PRESSURE" in reason
    assert "backing off" in reason
    assert "because" in budget.reasons  # pressure reasons carried through
