import sys
from types import SimpleNamespace, ModuleType

import pytest

from pharmacy_scraper.normalization.address import normalize_address
from pharmacy_scraper.normalization.phone import normalize_phone


@pytest.fixture
def fake_usaddress(monkeypatch):
    # Create a fake usaddress module with a tag function returning components
    m = ModuleType("usaddress")

    def tag(addr_str):
        # Return dict-like result similar to usaddress.tag
        # Example input: "123 Main St, Springfield, IL 62704"
        return (
            {
                "AddressNumber": "123",
                "StreetName": "Main",
                "StreetNamePostType": "St",
                "PlaceName": "Springfield",
                "StateName": "IL",
                "ZipCode": "62704",
            },
            "Street Address",
        )

    m.tag = tag
    monkeypatch.setitem(sys.modules, "usaddress", m)
    return m


@pytest.fixture
def fake_libpostal(monkeypatch):
    # Create a fake postal.parser module with parse_address
    postal = ModuleType("postal")
    parser = ModuleType("parser")

    def parse_address(addr_str):
        # Return list of tuples (component, label)
        # Example for a Canadian address
        return [
            ("100 King St W", "house"),
            ("Toronto", "city"),
            ("ON", "state"),
            ("M5X 1A9", "postcode"),
            ("ca", "country"),
        ]

    parser.parse_address = parse_address
    postal.parser = parser
    monkeypatch.setitem(sys.modules, "postal", postal)
    monkeypatch.setitem(sys.modules, "postal.parser", parser)
    return postal


def test_usaddress_parses_us_address_when_available(fake_usaddress):
    cfg = SimpleNamespace(INTERNATIONAL_ENABLED=0)
    out = normalize_address("123 Main St, Springfield, IL 62704", config=cfg)
    assert out["address_line1"] == "123 Main St"
    assert out["city"] == "Springfield"
    assert out["state"] == "IL"
    assert out["postal_code"] == "62704"
    assert out["country_iso2"] == "US"


def test_libpostal_parses_international_when_enabled(fake_libpostal):
    cfg = SimpleNamespace(INTERNATIONAL_ENABLED=1)
    out = normalize_address("100 King St W, Toronto, ON M5X 1A9, Canada", config=cfg)
    assert out["address_line1"].startswith("100 King St")
    assert out["city"] == "Toronto"
    assert out["state"] == "ON"
    assert out["postal_code"].startswith("M5X")
    assert out["country_iso2"].upper() in ("CA", "CAN")
    assert out.get("country_code") in ("CA", "ca")


def test_phonenumbers_used_when_available(monkeypatch):
    # Provide a fake phonenumbers with expected interface
    m = ModuleType("phonenumbers")

    class FakeNum:
        pass

    def parse(raw, region=None):
        return FakeNum()

    class Fmt:
        E164 = 0
        NATIONAL = 1

    def is_possible_number(num):
        return True

    def is_valid_number(num):
        return True

    def format_number(num, fmt):
        if fmt == Fmt.E164:
            return "+14155552671"
        return "(415) 555-2671"

    m.parse = parse
    m.PhoneNumberFormat = Fmt
    m.is_possible_number = is_possible_number
    m.is_valid_number = is_valid_number
    m.format_number = format_number

    monkeypatch.setitem(sys.modules, "phonenumbers", m)

    cfg = SimpleNamespace(default_region="US")
    out = normalize_phone("(415) 555-2671", config=cfg)
    assert out["phone_e164"] == "+14155552671"
    assert out["phone_national"] == "(415) 555-2671"
