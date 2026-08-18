"""Configuration file support.

Only knobs that already exist are exposed — the spec's warning against exposing
dozens of parameters prematurely is easier to obey than to undo. Precedence is
CLI over file over defaults.
"""

import logging
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

CONFIG_PATH = Path.home() / ".adaptive_compute" / "config.yaml"


@dataclass(frozen=True)
class Config:
    policy: str = "threshold"
    fraction: float = 0.5  # used by the fixed policy
    interval: float = 1.0  # telemetry sampling interval, seconds
    period: float = 1.0  # duty-cycle period, seconds
    nice: int = 0
    grace: float = 15.0  # seconds between SIGTERM and SIGKILL
    probe: bool = True

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        path = path or CONFIG_PATH
        try:
            text = path.read_text()
        except FileNotFoundError:
            return cls()
        except OSError:
            log.warning("could not read config %s; using defaults", path, exc_info=True)
            return cls()
        return cls.from_mapping(_parse_yaml(text), source=str(path))

    @classmethod
    def from_mapping(cls, data: dict[str, Any], source: str = "config") -> "Config":
        known = {f.name for f in fields(cls)}
        flat = _flatten(data)
        unknown = flat.keys() - known
        if unknown:
            log.warning("ignoring unknown settings in %s: %s", source, ", ".join(sorted(unknown)))
        return cls(**{k: v for k, v in flat.items() if k in known})

    def merged_with_cli(self, args: Any) -> "Config":
        """CLI options win where supplied.

        Relies on the parser defaulting these options to None, so "not given"
        is distinguishable from "given the same value as the default".
        """
        overrides = {
            f.name: getattr(args, f.name)
            for f in fields(self)
            if getattr(args, f.name, None) is not None
        }
        return type(self)(**{**self.__dict__, **overrides})


def _flatten(data: dict[str, Any]) -> dict[str, Any]:
    """Accept both flat keys and the spec's nested sections."""
    flat: dict[str, Any] = {}
    for key, value in (data or {}).items():
        if isinstance(value, dict):
            flat.update(value)
        else:
            flat[key] = value
    return flat


def _parse_yaml(text: str) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        log.warning("PyYAML is not installed; config file ignored")
        return {}
    parsed = yaml.safe_load(text)
    if not isinstance(parsed, dict):
        log.warning("config file is not a mapping; ignoring it")
        return {}
    return parsed
