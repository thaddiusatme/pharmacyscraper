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

    # Simple Canadian heuristic for tests
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
