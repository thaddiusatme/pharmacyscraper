from types import SimpleNamespace

import pytest

# TDD: write tests first; functions to be implemented
from pharmacy_scraper.normalization.phone import normalize_phone


def test_us_phone_formats_to_e164_and_national_default_us():
    raw = "(415) 555-2671"
    cfg = SimpleNamespace(default_region="US")
    out = normalize_phone(raw, config=cfg)
    assert out["phone_e164"].startswith("+") and len(out["phone_e164"]) > 5
    assert "415" in out["phone_national"]


def test_non_us_phone_requires_international_enabled_or_region():
    raw = "+44 20 7946 0958"
    # Without specifying region or international flag, should still parse due to E.164 but keep policy simple
    cfg = SimpleNamespace(default_region=None, INTERNATIONAL_ENABLED=0)
    out = normalize_phone(raw, config=cfg)
    assert out["phone_e164"].startswith("+44")
    assert out["phone_national"]
