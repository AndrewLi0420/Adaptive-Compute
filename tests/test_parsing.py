from adaptive_compute.platform.macos.gpu import parse_gpu_utilization
from adaptive_compute.platform.macos.idle import parse_hid_idle_seconds
from adaptive_compute.platform.macos.pressure import pressure_level_name
from adaptive_compute.platform.macos.thermal import thermal_state_name

IOREG_HID_FIXTURE = """\
+-o IOHIDSystem  <class IOHIDSystem, id 0x100000466, registered, matched, active, busy 0 (2 ms), retain 24>
    {
      "IOClass" = "IOHIDSystem"
      "HIDIdleTime" = 9607690000
      "IOProviderClass" = "IOResources"
    }
"""

IOREG_GPU_FIXTURE = """\
+-o AGXAcceleratorG15X_B0  <class AGXAcceleratorG15X_B0, id 0x1000004d1, registered, matched, active, busy 0 (13 ms), retain 40>
    {
      "PerformanceStatistics" = {"Device Utilization %"=86,"Renderer Utilization %"=80,"Tiler Utilization %"=12}
      "IOClass" = "AGXAcceleratorG15X_B0"
    }
"""


def test_parse_hid_idle():
    assert parse_hid_idle_seconds(IOREG_HID_FIXTURE) == 9.60769


def test_parse_hid_idle_missing():
    assert parse_hid_idle_seconds("no such key here") is None


def test_parse_gpu_utilization():
    assert parse_gpu_utilization(IOREG_GPU_FIXTURE) == 86.0


def test_parse_gpu_utilization_missing():
    assert parse_gpu_utilization("") is None


def test_pressure_level_names():
    assert pressure_level_name(1) == "normal"
    assert pressure_level_name(2) == "warn"
    assert pressure_level_name(4) == "critical"
    assert pressure_level_name(3) == "unknown(3)"


def test_thermal_state_names():
    assert thermal_state_name(0) == "nominal"
    assert thermal_state_name(3) == "critical"
    assert thermal_state_name(9) == "unknown(9)"
