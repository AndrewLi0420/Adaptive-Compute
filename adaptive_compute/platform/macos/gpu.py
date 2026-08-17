import re
import subprocess
from typing import Any

# "Device Utilization %" comes from the IOAccelerator registry entry's
# PerformanceStatistics dict. Undocumented interface: works on current
# Apple Silicon but may disappear in any macOS release, hence best-effort.
_GPU_UTIL_RE = re.compile(r'"Device Utilization %"\s*=\s*(\d+)')


def parse_gpu_utilization(ioreg_output: str) -> float | None:
    match = _GPU_UTIL_RE.search(ioreg_output)
    if match is None:
        return None
    return float(match.group(1))


class GpuProvider:
    name = "gpu"

    def sample(self) -> dict[str, Any]:
        out = subprocess.run(
            ["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        return {"gpu_utilization": parse_gpu_utilization(out)}
