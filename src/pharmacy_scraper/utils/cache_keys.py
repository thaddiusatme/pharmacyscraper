from __future__ import annotations

from typing import Any, Dict, Union


def _norm(text: str) -> str:
    return text.lower().strip() if isinstance(text, str) else ""


def pharmacy_cache_key(pharmacy: Union[Dict[str, Any], Any], *, use_llm: bool = True) -> str:
    """Deterministic cache key for a pharmacy classification result.

    Accepts dicts or objects with attributes `name` and `address`.
    """
    name = ""
    address = ""

    if isinstance(pharmacy, dict):
        name = _norm(pharmacy.get("name"))
        address = _norm(pharmacy.get("address"))
    else:
        name = _norm(getattr(pharmacy, "name", ""))
        address = _norm(getattr(pharmacy, "address", ""))

    return f"{name}:{address}:{use_llm}"
