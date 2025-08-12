import json
import tempfile
from pathlib import Path

from pharmacy_scraper.config.loader import load_config


def write_tmp_config(data: dict) -> str:
    fd = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    path = Path(fd.name)
    fd.close()
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def test_feature_flags_default_zero_when_missing():
    cfg_path = write_tmp_config({})
    cfg = load_config(cfg_path)

    assert cfg["EMAIL_DISCOVERY_ENABLED"] == 0
    assert cfg["INTERNATIONAL_ENABLED"] == 0


def test_feature_flags_accept_bool_and_int():
    cfg_path = write_tmp_config({
        "EMAIL_DISCOVERY_ENABLED": True,
        "INTERNATIONAL_ENABLED": 1,
    })
    cfg = load_config(cfg_path)

    assert cfg["EMAIL_DISCOVERY_ENABLED"] == 1
    assert cfg["INTERNATIONAL_ENABLED"] == 1
