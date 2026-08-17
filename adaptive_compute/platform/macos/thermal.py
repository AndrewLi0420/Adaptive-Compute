import ctypes
import ctypes.util
from typing import Any

# NSProcessInfoThermalState
THERMAL_STATES = {0: "nominal", 1: "fair", 2: "serious", 3: "critical"}


def thermal_state_name(state: int) -> str:
    return THERMAL_STATES.get(state, f"unknown({state})")


class _ObjCProcessInfo:
    """Minimal Objective-C bridge to [NSProcessInfo processInfo]."""

    def __init__(self):
        self._objc = ctypes.CDLL(ctypes.util.find_library("objc"))
        # Foundation must be loaded for the NSProcessInfo class to exist
        ctypes.CDLL(ctypes.util.find_library("Foundation"))
        self._objc.objc_getClass.restype = ctypes.c_void_p
        self._objc.sel_registerName.restype = ctypes.c_void_p
        msg_ptr = ctypes.cast(
            self._objc.objc_msgSend,
            ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p),
        )
        self._msg_long = ctypes.cast(
            self._objc.objc_msgSend,
            ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p),
        )
        cls = self._objc.objc_getClass(b"NSProcessInfo")
        self._info = msg_ptr(cls, self._objc.sel_registerName(b"processInfo"))
        self._sel_thermal = self._objc.sel_registerName(b"thermalState")
        self._sel_lowpower = self._objc.sel_registerName(b"isLowPowerModeEnabled")

    def thermal_state(self) -> int:
        return self._msg_long(self._info, self._sel_thermal)

    def low_power_mode(self) -> bool:
        return bool(self._msg_long(self._info, self._sel_lowpower))


class ThermalProvider:
    """Thermal state via NSProcessInfo (supported API, coarse 4-level enum)."""

    name = "thermal"

    def __init__(self):
        self._info = _ObjCProcessInfo()

    def sample(self) -> dict[str, Any]:
        return {
            "thermal_state": thermal_state_name(self._info.thermal_state()),
            "low_power_mode": self._info.low_power_mode(),
        }
