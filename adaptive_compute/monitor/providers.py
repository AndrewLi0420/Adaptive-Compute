from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Provider(Protocol):
    """A telemetry source that fills some subset of SystemState fields.

    sample() returns a mapping of SystemState field names to values. It may
    raise; the Sampler isolates failures so one bad source never spoils a
    sample. Providers are called from a single sampling thread and do not
    need to be thread-safe.
    """

    name: str

    def sample(self) -> dict[str, Any]: ...
