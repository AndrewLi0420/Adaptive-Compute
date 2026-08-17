from adaptive_compute.monitor.providers import Provider
from adaptive_compute.platform.macos.cpu_mem import CpuMemProvider
from adaptive_compute.platform.macos.gpu import GpuProvider
from adaptive_compute.platform.macos.idle import IdleProvider
from adaptive_compute.platform.macos.power import PowerProvider
from adaptive_compute.platform.macos.pressure import MemoryPressureProvider
from adaptive_compute.platform.macos.thermal import ThermalProvider


def default_providers(pid: int | None = None) -> list[Provider]:
    return [
        CpuMemProvider(pid=pid),
        MemoryPressureProvider(),
        ThermalProvider(),
        PowerProvider(),
        IdleProvider(),
        GpuProvider(),
    ]
