import re
import subprocess
from typing import Any

# HIDIdleTime is nanoseconds since the last human input event (any keyboard/
# mouse/trackpad event resets it). We read only elapsed time, never content.
_HID_IDLE_RE = re.compile(r'"HIDIdleTime"\s*=\s*(\d+)')


def parse_hid_idle_seconds(ioreg_output: str) -> float | None:
    match = _HID_IDLE_RE.search(ioreg_output)
    if match is None:
        return None
    return int(match.group(1)) / 1e9


class IdleProvider:
    name = "idle"

    def sample(self) -> dict[str, Any]:
        out = subprocess.run(
            ["ioreg", "-r", "-d", "1", "-k", "HIDIdleTime", "-c", "IOHIDSystem"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        return {"user_idle_seconds": parse_hid_idle_seconds(out)}
