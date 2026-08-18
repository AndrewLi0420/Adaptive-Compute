from dataclasses import dataclass, fields


@dataclass(frozen=True, slots=True)
class SystemState:
    """One sample of system telemetry.

    Every field except `timestamp` is optional: a provider that fails or a
    metric the platform does not expose is represented as None, never guessed.
    Percentages are 0-100.
    """

    timestamp: float

    cpu_utilization: float | None = None
    per_core_utilization: list[float] | None = None
    load_avg_1m: float | None = None
    cpu_count_logical: int | None = None
    cpu_count_physical: int | None = None

    memory_total_bytes: int | None = None
    memory_available_bytes: int | None = None
    memory_utilization: float | None = None
    swap_used_bytes: int | None = None
    memory_pressure: str | None = None  # normal | warn | critical

    process_cpu_percent: float | None = None
    process_memory_bytes: int | None = None

    thermal_state: str | None = None  # nominal | fair | serious | critical
    low_power_mode: bool | None = None

    plugged_in: bool | None = None
    battery_percent: float | None = None

    user_idle_seconds: float | None = None

    gpu_utilization: float | None = None  # best-effort, undocumented source

    # Responsiveness percentiles over the probe's recent window; the headline
    # field is p95. Latency = scheduler wakeup delay + small work unit.
    responsiveness_latency_ms: float | None = None
    responsiveness_p50_ms: float | None = None
    responsiveness_p99_ms: float | None = None

    monitor_overhead_ms: float | None = None


SYSTEM_STATE_FIELDS = frozenset(f.name for f in fields(SystemState))
