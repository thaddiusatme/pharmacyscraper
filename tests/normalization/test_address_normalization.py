from types import SimpleNamespace

import pytest

# TDD: write tests first; functions to be implemented
from pharmacy_scraper.normalization.address import normalize_address


def test_us_address_is_parsed_with_us_defaults():
    raw = "123 Main St, Springfield, IL 62704"
    cfg = SimpleNamespace(INTERNATIONAL_ENABLED=0)
    out = normalize_address(raw, config=cfg)
    assert out["address_line1"] == "123 Main St"
    assert out["city"] == "Springfield"
    assert out["state"] == "IL"
    assert out["postal_code"] == "62704"
    assert out["country_iso2"] == "US"


def test_international_disabled_returns_minimal_for_non_us():
    raw = "100 King St W, Toronto, ON M5X 1A9, Canada"
    cfg = SimpleNamespace(INTERNATIONAL_ENABLED=0)
    out = normalize_address(raw, config=cfg)
    # when international disabled, we do not attempt non-US parsing; ensure safe defaults
    assert out["country_iso2"] in (None, "US")


def test_international_enabled_includes_country_code():
    raw = "100 King St W, Toronto, ON M5X 1A9, Canada"
    cfg = SimpleNamespace(INTERNATIONAL_ENABLED=1)
    out = normalize_address(raw, config=cfg)
    # Expect country inference when international enabled
    assert out["country_iso2"] in ("CA", "CAN", "CA ") or out.get("country_code") in ("CA",)
