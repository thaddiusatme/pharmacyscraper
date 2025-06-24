"""classifier.py
Clean and minimal implementation of pharmacy classification utilities required by
unit tests. No external network calls are made here.

Public API expected by the tests:
- CHAIN_IDENTIFIERS (list[str])
- rule_based_classify(pharmacy: dict) -> dict
- query_perplexity(pharmacy: dict) -> dict    # stub; tests monkey-patch
- classify_pharmacy(pharmacy: dict, *, use_llm=True) -> dict
- batch_classify_pharmacies(pharmacies: list[dict], **kwargs) -> list[dict]
- Classifier class with method classify_pharmacy
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

###############################################################################
# Constant
###############################################################################

CHAIN_IDENTIFIERS: List[str] = [
    # Major US retail chains
    "CVS",
    "Walgreens",
    "Rite Aid",
    "Walmart",
    "Costco",
    "Kroger",
    "Safeway",
    "Albertsons",
    "Publix",
    "Target",
    "Sam's Club",
    "Meijer",
    "H-E-B",
    "Fred Meyer",
    "Hy-Vee",
    "Wegmans",
    "Giant",
    "Stop & Shop",
    # Hospital / health-system indicators
    "Hospital",
    "Clinic",
    "VA",
    "Medical Center",
    "Health System",
    "Kaiser",
    "ANMC",
    "ANTHC",
]

###############################################################################
# Simple helpers
###############################################################################

def _norm(text: str) -> str:
    """Lower-case helper that safely handles non-string input."""
    return text.lower() if isinstance(text, str) else ""


def rule_based_classify(pharmacy: Dict) -> Dict:
    """Keyword heuristic classification.

    Required keys produced: is_chain, is_compounding, confidence, reason,
    method, source.
    """
    name = _norm(pharmacy.get("name") or pharmacy.get("title", ""))

    # 1. Chain / hospital detection
    import re

    def _token_match(text: str, keyword: str) -> bool:
        pattern = rf"\b{re.escape(keyword.lower())}\b"
        return re.search(pattern, text) is not None

    for kw in CHAIN_IDENTIFIERS:
        if _token_match(name, kw):
            return {
                "is_chain": True,
                "is_compounding": False,
                "confidence": 1.0,
                "reason": f"Matched chain keyword: {kw}",
                "method": "rule_based",
                "source": "rule-based",
            }

    # 2. Compounding pharmacies
    if "compounding" in name:
        return {
            "is_chain": False,
            "is_compounding": True,
            "confidence": 0.95,
            "reason": "Compounding pharmacy keyword detected",
            "method": "rule_based",
            "source": "rule-based",
        }

    # 3. Default independent
    return {
        "is_chain": False,
        "is_compounding": False,
        "confidence": 0.5,
        "reason": "No chain identifiers",
        "method": "rule_based",
        "source": "rule-based",
    }


def query_perplexity(pharmacy: Dict) -> Dict:
    """Stub LLM call. Tests usually monkey-patch this function."""
    logger.debug("query_perplexity stub called; returning fixed result")
    return {
        "is_chain": False,
        "is_compounding": False,
        "confidence": 0.75,
        "reason": "Stub LLM result",
        "method": "llm",
        "source": "perplexity",
    }


def classify_pharmacy(pharmacy: Dict, *, use_llm: bool = True) -> Dict:
    """Orchestrate rule-based + optional LLM; return higher-confidence."""
    rule_res = rule_based_classify(pharmacy)
    if not use_llm or rule_res["confidence"] >= 0.9:
        return rule_res
    llm_res = query_perplexity(pharmacy)
    return llm_res if llm_res["confidence"] >= rule_res["confidence"] else rule_res


def batch_classify_pharmacies(pharmacies: List[Dict], **kwargs) -> List[Dict]:
    """Apply classify_pharmacy to each input dict."""
    return [classify_pharmacy(p, **kwargs) for p in pharmacies]


###############################################################################
# Class interface
###############################################################################

class Classifier:
    """Wrapper utilised in some tests where a PerplexityClient mock is injected."""

    def __init__(self, client: Optional["PerplexityClient"] = None):
        if client is None:
            try:
                # Lazy import so tests can patch before import
                from .perplexity_client import PerplexityClient  # type: ignore

                client = PerplexityClient()
            except Exception:
                client = None  # pragma: no cover
        self._client = client

    def classify_pharmacy(self, pharmacy: Dict) -> Dict:
        """Classifies a single pharmacy.

        Args:
            pharmacy: Dictionary with pharmacy data.

        Returns:
            Dictionary with classification results.
        """
        # Try cheap rule first
        res = rule_based_classify(pharmacy)
        if res["confidence"] >= 0.9 or res.get("is_compounding"):
            return res
        
        # Else delegate to client if available
        if self._client is not None:
            try:
                llm_res = self._client.classify_pharmacy(pharmacy)  # type: ignore[attr-defined]
                # Convert from the client's format to our expected format
                return {
                    "is_chain": llm_res.get("classification") == "chain",
                    "is_compounding": llm_res.get("is_compounding", False),
                    "confidence": llm_res.get("confidence", 0.0),
                    "reason": f"LLM classification: {llm_res.get('classification', 'unknown')}",
                    "method": "llm",
                    "source": "perplexity",
                }
            except Exception as e:
                logger.error(f"LLM classification failed: {e}")
                # Fall through to stub
        
        # Fallback stub
        return query_perplexity(pharmacy)


###############################################################################
# Re-export list
###############################################################################

__all__ = [
    "CHAIN_IDENTIFIERS",
    "rule_based_classify",
    "query_perplexity",
    "classify_pharmacy",
    "batch_classify_pharmacies",
    "Classifier",
]