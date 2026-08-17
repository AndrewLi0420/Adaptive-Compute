import ctypes
from typing import Any

# kern.memorystatus_vm_pressure_level values (XNU kern_memorystatus.h)
PRESSURE_LEVELS = {1: "normal", 2: "warn", 4: "critical"}

_libc = ctypes.CDLL(None, use_errno=True)


def sysctl_int(name: str) -> int:
    val = ctypes.c_int()
    size = ctypes.c_size_t(ctypes.sizeof(val))
    rc = _libc.sysctlbyname(name.encode(), ctypes.byref(val), ctypes.byref(size), None, 0)
    if rc != 0:
        raise OSError(ctypes.get_errno(), f"sysctlbyname({name}) failed")
    return val.value


def pressure_level_name(level: int) -> str:
    return PRESSURE_LEVELS.get(level, f"unknown({level})")


class MemoryPressureProvider:
    """Kernel memory pressure level. Undocumented but long-stable sysctl;
    preferred over computing utilization ourselves because it accounts for
    the memory compressor."""

    name = "memory_pressure"

    def sample(self) -> dict[str, Any]:
        level = sysctl_int("kern.memorystatus_vm_pressure_level")
        return {"memory_pressure": pressure_level_name(level)}
