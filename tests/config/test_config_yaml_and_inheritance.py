import os
import json
from pathlib import Path
import pytest


def write_yaml(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(text)
    return p


def test_loads_yaml_with_env_substitution(tmp_path, monkeypatch):
    from pharmacy_scraper.config.loader import load_config

    monkeypatch.setenv("OUTDIR", "outdir_env")
    yaml_text = """
    output_dir: ${OUTDIR}
    cache_dir: cache
    api_keys:
      apify: ${APIFY_TOKEN:-fallback}
    """
    path = write_yaml(tmp_path, yaml_text)
    cfg = load_config(str(path))
    assert cfg["output_dir"] == "outdir_env"
    assert cfg["api_keys"]["apify"] == "fallback"


def test_stricter_schema_rejects_unknown_top_level_keys(tmp_path):
    from pharmacy_scraper.config.loader import load_config

    data = {"unknown_key": True}
    p = tmp_path / "config.json"
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_config(str(p))


def test_env_inheritance_merges_overrides(tmp_path):
    from pharmacy_scraper.config.loader import load_config

    yaml_text = """
    output_dir: output
    cache_dir: cache
    api_keys: {apify: base}
    env: dev
    environments:
      dev:
        output_dir: dev_output
        plugin_mode: true
        plugins:
          sources: [pharmacy_scraper.plugins.csv_source.CSVSourcePlugin]
        plugin_config:
          CSVSourcePlugin: {path: data.csv}
    """
    path = write_yaml(tmp_path, yaml_text)
    cfg = load_config(str(path))
    assert cfg["output_dir"] == "dev_output"
    assert cfg["plugin_mode"] is True
    assert cfg["plugins"]["sources"]
    assert cfg["plugin_config"]["CSVSourcePlugin"]["path"] == "data.csv"


def test_env_inheritance_missing_env_raises(tmp_path):
    from pharmacy_scraper.config.loader import load_config

    yaml_text = """
    output_dir: output
    env: staging
    environments:
      dev: {output_dir: dev}
    """
    path = write_yaml(tmp_path, yaml_text)
    with pytest.raises(ValueError):
        load_config(str(path))
