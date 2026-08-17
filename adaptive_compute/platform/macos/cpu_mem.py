from typing import Any

import psutil


class CpuMemProvider:
    """CPU, memory, swap, and (optionally) tracked-process metrics via psutil.

    cpu_percent() measures utilization since the previous call, so the first
    sample after construction reports the interval since __init__ (primed
    with a throwaway call there).
    """

    name = "cpu_mem"

    def __init__(self, pid: int | None = None):
        self._process: psutil.Process | None = None
        if pid is not None:
            self._process = psutil.Process(pid)
            self._process.cpu_percent()
        psutil.cpu_percent(percpu=True)

    def sample(self) -> dict[str, Any]:
        per_core = psutil.cpu_percent(percpu=True)
        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()
        result: dict[str, Any] = {
            "cpu_utilization": sum(per_core) / len(per_core),
            "per_core_utilization": per_core,
            "load_avg_1m": psutil.getloadavg()[0],
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "memory_total_bytes": vm.total,
            "memory_available_bytes": vm.available,
            "memory_utilization": vm.percent,
            "swap_used_bytes": swap.used,
        }
        if self._process is not None:
            result.update(self._sample_process())
        return result

    def _sample_process(self) -> dict[str, Any]:
        assert self._process is not None
        try:
            cpu = self._process.cpu_percent()
            mem = self._process.memory_info().rss
            for child in self._process.children(recursive=True):
                try:
                    cpu += child.cpu_percent()
                    mem += child.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            return {"process_cpu_percent": cpu, "process_memory_bytes": mem}
        except psutil.NoSuchProcess:
            return {}
