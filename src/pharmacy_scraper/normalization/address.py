import importlib
import re
from typing import Dict, Optional


def normalize_address(raw: Optional[str], config=None) -> Dict[str, Optional[str]]:
    """Minimal address normalization to satisfy current TDD tests.

    - US-first handling: attempts to split common US formats and infer country_iso2="US"
    - International gating: when INTERNATIONAL_ENABLED=0, do not attempt non-US parsing
    - When INTERNATIONAL_ENABLED=1, infer Canada (CA) from simple heuristics and add country_code
    """
    out: Dict[str, Optional[str]] = {
        "address_line1": None,
        "address_line2": None,
        "city": None,
        "state": None,
        "postal_code": None,
        "country_iso2": None,
        # country_code may be added when INTERNATIONAL_ENABLED=1
    }

    if not raw or not isinstance(raw, str):
        return out

    text = raw.strip()
    intl_enabled = int(getattr(config, "INTERNATIONAL_ENABLED", 0)) if config else 0

    # Helper: try dynamic import so tests can monkeypatch sys.modules
    def _import(name: str):
        try:
            return importlib.import_module(name)
        except Exception:
            return None

    # International path with libpostal if available
    if intl_enabled:
        postal = _import("postal.parser") or _import("postal")
        parse_address = None
        if postal is not None:
            # postal may be the parser module or have attribute parser
            if hasattr(postal, "parse_address"):
                parse_address = getattr(postal, "parse_address")
            elif hasattr(postal, "parser") and hasattr(postal.parser, "parse_address"):
                parse_address = getattr(postal.parser, "parse_address")
        if callable(parse_address):
            try:
                pairs = list(parse_address(text))
                known = {
                    "house",
                    "house_number",
                    "road",
                    "city",
                    "suburb",
                    "state",
                    "state_district",
                    "postcode",
                    "country",
                }
                components = {}
                for a, b in pairs:
                    if a in known:
                        components[a] = b
                    elif b in known:
                        components[b] = a
                # Map common libpostal components
                address_line1 = components.get("house") or components.get("road")
                if not address_line1:
                    # Compose from house_number + road
                    hn = components.get("house_number")
                    rd = components.get("road")
                    if hn or rd:
                        address_line1 = f"{hn or ''} {rd or ''}".strip()
                out["address_line1"] = address_line1 or out["address_line1"]
                out["city"] = components.get("city") or components.get("suburb") or out["city"]
                out["state"] = components.get("state") or components.get("state_district") or out["state"]
                out["postal_code"] = components.get("postcode") or out["postal_code"]
                country = components.get("country")
                if country:
                    cc = country.strip().upper()
                    # Best-effort 2-letter
                    if len(cc) > 2:
                        cc = cc[:2]
                    out["country_iso2"] = cc
                    out["country_code"] = cc
                return out
            except Exception:
                # fall through to heuristics
                pass

    # Simple Canadian heuristic for tests and fallback
    if intl_enabled and ("Canada" in text or ", CA" in text or text.endswith(", CA")):
        # try to parse city/state/postal by commas
        parts = [p.strip() for p in text.split(",")]
        if len(parts) >= 4:
            # e.g., 100 King St W, Toronto, ON M5X 1A9, Canada
            out["address_line1"] = parts[0]
            out["city"] = parts[1]
            # Split state + postal
            m = re.match(r"([A-Z]{2})\s+([A-Z0-9 ]+)$", parts[2])
            if m:
                out["state"] = m.group(1)
                out["postal_code"] = m.group(2).strip()
            out["country_iso2"] = "CA"
            out["country_code"] = "CA"
            return out
        # Fallback minimal
        out["country_iso2"] = "CA"
        out["country_code"] = "CA"
        return out

    # US parsing with usaddress if available
    usa = _import("usaddress")
    if usa is not None and hasattr(usa, "tag"):
        try:
            tagged, _ = usa.tag(text)
            num = tagged.get("AddressNumber")
            street = " ".join(filter(None, [tagged.get("StreetName"), tagged.get("StreetNamePostType")]))
            line1 = " ".join(filter(None, [num, street])) or tagged.get("StreetName")
            out["address_line1"] = line1 or out["address_line1"]
            out["city"] = tagged.get("PlaceName") or out["city"]
            out["state"] = tagged.get("StateName") or out["state"]
            out["postal_code"] = tagged.get("ZipCode") or out["postal_code"]
            out["country_iso2"] = "US"
            return out
        except Exception:
            pass

    # US simple pattern: "123 Main St, Springfield, IL 62704"
    us_parts = [p.strip() for p in text.split(",")]
    if len(us_parts) >= 3:
        out["address_line1"] = us_parts[0]
        out["city"] = us_parts[1]
        # Expect state and ZIP in the third part
        state_zip = us_parts[2]
        m = re.search(r"([A-Z]{2})\s+(\d{5}(?:-\d{4})?)", state_zip)
        if m:
            out["state"] = m.group(1)
            out["postal_code"] = m.group(2)
        out["country_iso2"] = "US"
        return out

    # If international disabled and not US format, return minimal defaults
    if not intl_enabled:
        return out

    return out
