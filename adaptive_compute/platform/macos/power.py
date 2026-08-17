from typing import Any

import psutil


class PowerProvider:
    name = "power"

    def sample(self) -> dict[str, Any]:
        battery = psutil.sensors_battery()
        if battery is None:
            return {}
        return {
            "plugged_in": battery.power_plugged,
            "battery_percent": battery.percent,
        }
