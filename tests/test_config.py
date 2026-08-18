import argparse

from adaptive_compute.config import Config


def args(**kwargs) -> argparse.Namespace:
    defaults = dict(policy=None, fraction=None, interval=None, period=None,
                    nice=None, grace=None, probe=None)
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_defaults_when_no_file(tmp_path):
    cfg = Config.load(tmp_path / "absent.yaml")
    assert cfg.policy == "threshold"
    assert cfg.interval == 1.0


def test_flat_and_nested_keys_both_work(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("policy: fixed\nscheduler:\n  fraction: 0.25\nlimits:\n  nice: 5\n")
    cfg = Config.load(path)
    assert cfg.policy == "fixed"
    assert cfg.fraction == 0.25
    assert cfg.nice == 5


def test_unknown_keys_are_ignored_with_a_warning(caplog):
    cfg = Config.from_mapping({"policy": "fixed", "max_flux_capacitors": 3})
    assert cfg.policy == "fixed"
    assert "max_flux_capacitors" in caplog.text


def test_malformed_file_falls_back_to_defaults(tmp_path, caplog):
    path = tmp_path / "config.yaml"
    path.write_text("just a string, not a mapping")
    assert Config.load(path).policy == "threshold"


def test_cli_overrides_file(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("policy: fixed\nfraction: 0.25\n")
    merged = Config.load(path).merged_with_cli(args(policy="unrestricted"))
    assert merged.policy == "unrestricted"
    assert merged.fraction == 0.25  # untouched by the CLI


def test_unsupplied_cli_options_do_not_override(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("interval: 5.0\n")
    assert Config.load(path).merged_with_cli(args()).interval == 5.0


def test_store_false_flag_overrides(tmp_path):
    assert Config().merged_with_cli(args(probe=False)).probe is False
