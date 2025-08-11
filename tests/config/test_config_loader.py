import json
import os
from pathlib import Path
import pytest


def write_json(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "config.json"
    p.write_text(json.dumps(data))
    return p


def test_valid_legacy_config_loads(tmp_path, monkeypatch):
    from pharmacy_scraper.config.loader import load_config

    cfg_in = {
        "output_dir": str(tmp_path / "out"),
        "cache_dir": str(tmp_path / "cache"),
        "api_keys": {"apify": "abc"},
        # no plugin_mode, defaults apply
    }
    cfg_path = write_json(tmp_path, cfg_in)

    cfg = load_config(str(cfg_path))
    assert cfg["output_dir"].endswith("out")
    assert cfg["cache_dir"].endswith("cache")
    assert cfg.get("verify_places") is True  # default applied


def test_plugin_mode_requires_types(tmp_path):
    from pharmacy_scraper.config.loader import load_config

    # invalid plugins type (should be dict)
    cfg_in = {
        "plugin_mode": True,
        "plugins": ["x"],
        "plugin_config": {},
        "api_keys": {},
    }
    cfg_path = write_json(tmp_path, cfg_in)
    with pytest.raises(ValueError):
        load_config(str(cfg_path))

    # invalid plugin_config type (should be dict)
    cfg_in2 = {
        "plugin_mode": True,
        "plugins": {"sources": []},
        "plugin_config": [],
        "api_keys": {},
    }
    cfg_path2 = write_json(tmp_path, cfg_in2)
    with pytest.raises(ValueError):
        load_config(str(cfg_path2))


def test_env_var_substitution_in_values(tmp_path, monkeypatch):
    from pharmacy_scraper.config.loader import load_config

    monkeypatch.setenv("APIFY_TOKEN", "secret123")
    cfg_in = {
        "api_keys": {"apify": "${APIFY_TOKEN}"},
        "output_dir": str(tmp_path / "out"),
        "cache_dir": "${UNSET_VAR:-cache}",
        "plugin_mode": True,
        "plugins": {"sources": []},
        "plugin_config": {},
    }
    cfg_path = write_json(tmp_path, cfg_in)

    cfg = load_config(str(cfg_path), env=os.environ)
    assert cfg["api_keys"]["apify"] == "secret123"
    assert cfg["cache_dir"].endswith("cache")


def test_defaults_applied_for_missing_optional_fields(tmp_path):
    from pharmacy_scraper.config.loader import load_config

    cfg_in = {"api_keys": {}}
    cfg_path = write_json(tmp_path, cfg_in)
    cfg = load_config(str(cfg_path))
    # defaults
    assert cfg["output_dir"] == "output"
    assert cfg["cache_dir"] == "cache"
    assert cfg["verify_places"] is True


def test_clear_errors_on_invalid_types_or_missing_required(tmp_path):
    from pharmacy_scraper.config.loader import load_config

    # output_dir wrong type
    cfg_in = {"output_dir": 123}
    cfg_path = write_json(tmp_path, cfg_in)
    with pytest.raises(ValueError):
        load_config(str(cfg_path))
