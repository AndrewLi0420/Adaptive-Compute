import pytest

from adaptive_compute.monitor.sampler import Sampler


class GoodProvider:
    name = "good"

    def sample(self):
        return {"cpu_utilization": 42.0}


class FailingProvider:
    name = "failing"

    def sample(self):
        raise RuntimeError("source unavailable")


class BadFieldProvider:
    name = "bad_field"

    def sample(self):
        return {"not_a_real_field": 1}


def test_failing_provider_is_isolated(caplog):
    sampler = Sampler([FailingProvider(), GoodProvider()])
    state = sampler.sample_once()
    assert state.cpu_utilization == 42.0
    assert state.memory_utilization is None
    assert "failing" in caplog.text


def test_unknown_field_raises():
    sampler = Sampler([BadFieldProvider()])
    with pytest.raises(ValueError, match="bad_field"):
        sampler.sample_once()


def test_overhead_is_measured():
    state = Sampler([GoodProvider()]).sample_once()
    assert state.monitor_overhead_ms is not None
    assert 0 <= state.monitor_overhead_ms < 1000


def test_run_stops(caplog):
    sampler = Sampler([GoodProvider()], interval_s=0.01)
    seen = []

    def on_sample(state):
        seen.append(state)
        if len(seen) >= 3:
            sampler.stop()

    sampler.run(on_sample)
    assert len(seen) == 3
