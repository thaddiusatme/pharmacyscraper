import json
import tempfile
from pathlib import Path

import pytest

from pharmacy_scraper.config.loader import load_config


def write_tmp_config(data: dict) -> str:
    fd = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    path = Path(fd.name)
    fd.close()
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def test_load_config_with_business_fields():
    cfg_path = write_tmp_config({
        "business_type": "pharmacy",
        "search_terms": ["independent pharmacy", "24 hour"],
    })

    cfg = load_config(cfg_path)

    assert cfg["business_type"] == "pharmacy"
    assert isinstance(cfg["search_terms"], list)
    assert cfg["search_terms"] == ["independent pharmacy", "24 hour"]


def test_business_type_defaults_pharmacy_when_missing():
    cfg_path = write_tmp_config({})

    cfg = load_config(cfg_path)

    assert cfg["business_type"] == "pharmacy"


def test_search_terms_defaults_to_empty_list_when_missing():
    cfg_path = write_tmp_config({
        "business_type": "pharmacy",
    })

    cfg = load_config(cfg_path)

    assert "search_terms" in cfg
    assert isinstance(cfg["search_terms"], list)
    assert cfg["search_terms"] == []
