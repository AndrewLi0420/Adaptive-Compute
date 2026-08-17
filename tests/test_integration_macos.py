import os
import sys
import time

import pytest

from adaptive_compute.monitor.sampler import Sampler
from adaptive_compute.platform.macos import default_providers

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")


def test_real_samples_have_sane_values():
    sampler = Sampler(default_providers(pid=os.getpid()))
    sampler.sample_once()
    time.sleep(0.2)  # give cpu_percent a real measurement interval
    state = sampler.sample_once()

    assert 0 <= state.cpu_utilization <= 100
    assert len(state.per_core_utilization) == state.cpu_count_logical
    assert all(0 <= c <= 100 for c in state.per_core_utilization)

    assert state.memory_total_bytes > 0
    assert 0 < state.memory_available_bytes < state.memory_total_bytes
    assert 0 <= state.memory_utilization <= 100
    assert state.swap_used_bytes >= 0
    assert state.memory_pressure in ("normal", "warn", "critical")

    assert state.thermal_state in ("nominal", "fair", "serious", "critical")
    assert isinstance(state.low_power_mode, bool)

    assert state.process_cpu_percent is not None
    assert state.process_memory_bytes > 0

    if state.user_idle_seconds is not None:
        assert state.user_idle_seconds >= 0
    if state.gpu_utilization is not None:
        assert 0 <= state.gpu_utilization <= 100
    if state.battery_percent is not None:
        assert 0 <= state.battery_percent <= 100

    assert state.monitor_overhead_ms < 500
