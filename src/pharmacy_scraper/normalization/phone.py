from typing import Dict, Optional

try:
    import phonenumbers
except Exception:  # pragma: no cover - allow tests to run without dependency
    phonenumbers = None


def normalize_phone(raw: Optional[str], config=None) -> Dict[str, Optional[str]]:
    """Minimal phone normalization to satisfy current TDD tests.
    Attempts to parse using phonenumbers when available; otherwise performs basic sanitization.
    Returns dict with keys: phone_e164, phone_national.
    """
    out = {"phone_e164": None, "phone_national": None}
    if not raw or not isinstance(raw, str):
        return out

    default_region = getattr(config, "default_region", "US") if config else "US"

    # If phonenumbers is present, use it for robust parsing.
    if phonenumbers is not None:
        try:
            num = phonenumbers.parse(raw, default_region or None)
            if phonenumbers.is_possible_number(num) or phonenumbers.is_valid_number(num):
                out["phone_e164"] = phonenumbers.format_number(num, phonenumbers.PhoneNumberFormat.E164)
                out["phone_national"] = phonenumbers.format_number(num, phonenumbers.PhoneNumberFormat.NATIONAL)
                return out
        except Exception:
            pass

    # Fallback: simple sanitize
    digits = "".join(ch for ch in raw if ch.isdigit() or ch == "+")
    if digits.startswith("+"):
        out["phone_e164"] = digits
    else:
        # Assume US for fallback
        out["phone_e164"] = "+1" + "".join(ch for ch in raw if ch.isdigit())
    # National: keep area code in simple form
    cleaned = "".join(ch for ch in raw if ch.isdigit())
    if len(cleaned) >= 10:
        out["phone_national"] = f"({cleaned[-10:-7]}) {cleaned[-7:-4]}-{cleaned[-4:]}"
    else:
        out["phone_national"] = cleaned
    return out
