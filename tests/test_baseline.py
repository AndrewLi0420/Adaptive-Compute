import json
import platform

from adaptive_compute.monitor.baseline import Baseline, load_baseline, save_baseline
from adaptive_compute.monitor.probe import ProbeStats

STATS = ProbeStats(
    p50_ms=0.12, p95_ms=0.18, p99_ms=0.9, wake_p95_ms=5.05, sample_count=300
)


def test_round_trip(tmp_path):
    path = tmp_path / "baseline.json"
    save_baseline(Baseline.from_stats(STATS), path)
    loaded = load_baseline(path)
    assert loaded.p95_ms == 0.18
    assert loaded.wake_p95_ms == 5.05
    assert loaded.sample_count == 300
    assert loaded.python_version == platform.python_version()
    assert loaded.recorded_at > 0


def test_missing_file_returns_none(tmp_path):
    assert load_baseline(tmp_path / "absent.json") is None


def test_corrupt_file_returns_none(tmp_path, caplog):
    path = tmp_path / "baseline.json"
    path.write_text("{not json")
    assert load_baseline(path) is None
    assert "unreadable" in caplog.text


def test_unexpected_schema_returns_none(tmp_path, caplog):
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({"totally": "different"}))
    assert load_baseline(path) is None
    assert "unexpected fields" in caplog.text


def test_python_version_mismatch_warns(tmp_path, caplog):
    path = tmp_path / "baseline.json"
    save_baseline(Baseline.from_stats(STATS), path)
    raw = json.loads(path.read_text())
    raw["python_version"] = "3.0.0"
    path.write_text(json.dumps(raw))
    assert load_baseline(path) is not None  # still usable, but flagged
    assert "re-run" in caplog.text
