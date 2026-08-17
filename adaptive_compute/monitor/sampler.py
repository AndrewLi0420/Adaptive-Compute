import logging
import threading
import time
from typing import Any, Callable

from adaptive_compute.monitor.providers import Provider
from adaptive_compute.monitor.state import SYSTEM_STATE_FIELDS, SystemState

log = logging.getLogger(__name__)


class Sampler:
    """Assembles SystemState samples from a set of providers.

    Single-threaded: sample_once() and run() must be called from one thread.
    A provider that raises is logged and its fields stay None for that sample.
    """

    def __init__(self, providers: list[Provider], interval_s: float = 1.0):
        self.providers = providers
        self.interval_s = interval_s
        self._stop = threading.Event()

    def sample_once(self) -> SystemState:
        start = time.perf_counter()
        merged: dict[str, Any] = {}
        for provider in self.providers:
            try:
                result = provider.sample()
            except Exception:
                log.warning("provider %s failed", provider.name, exc_info=True)
                continue
            unknown = result.keys() - SYSTEM_STATE_FIELDS
            if unknown:
                raise ValueError(f"provider {provider.name} returned unknown fields: {unknown}")
            merged.update(result)
        overhead_ms = (time.perf_counter() - start) * 1000
        return SystemState(timestamp=time.time(), monitor_overhead_ms=overhead_ms, **merged)

    def run(self, callback: Callable[[SystemState], None]) -> None:
        """Sample every interval_s until stop(), correcting for drift."""
        next_at = time.monotonic()
        while not self._stop.is_set():
            callback(self.sample_once())
            next_at += self.interval_s
            delay = next_at - time.monotonic()
            if delay < 0:
                # sampling fell behind; skip missed ticks rather than bursting
                next_at = time.monotonic() + self.interval_s
                delay = self.interval_s
            self._stop.wait(delay)

    def stop(self) -> None:
        self._stop.set()
