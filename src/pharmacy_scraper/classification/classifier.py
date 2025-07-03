"""classifier.py
Clean and minimal implementation of pharmacy classification utilities.

This module provides classification functionality using both rule-based and LLM-based
approaches, with support for caching and batch processing.

Public API:
- CHAIN_IDENTIFIERS (list[str]): List of known chain pharmacy identifiers
- rule_based_classify(pharmacy: Union[Dict, PharmacyData]) -> ClassificationResult
- query_perplexity(pharmacy: Union[Dict, PharmacyData]) -> ClassificationResult
- classify_pharmacy(pharmacy: Union[Dict, PharmacyData], *, use_llm=True) -> ClassificationResult
- batch_classify_pharmacies(pharmacies: List[Union[Dict, PharmacyData]], **kwargs) -> List[ClassificationResult]
- Classifier: Class for classifying pharmacies with optional LLM fallback
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Union, Any, cast

from .models import (
    PharmacyData,
    ClassificationResult,
    ClassificationMethod,
    ClassificationSource,
)

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


def rule_based_classify(pharmacy: Union[Dict, PharmacyData]) -> ClassificationResult:
    """Classify a pharmacy using rule-based heuristics.
    
    Args:
        pharmacy: Pharmacy data as either a dictionary or PharmacyData instance
        
    Returns:
        ClassificationResult with the classification results
    """
    # Convert dict to PharmacyData if needed
    if isinstance(pharmacy, dict):
        pharmacy_data = PharmacyData.from_dict(pharmacy)
    else:
        pharmacy_data = pharmacy
        
    # Set default classification to 'independent' if not specified
    classification = "independent"
    
    name = _norm(pharmacy_data.name or "")

    # 1. Chain / hospital detection
    import re

    def _token_match(text: str, keyword: str) -> bool:
        pattern = rf"\b{re.escape(keyword.lower())}\b"
        return re.search(pattern, text) is not None

    # Check for chain pharmacies
    for kw in CHAIN_IDENTIFIERS:
        if _token_match(name, kw):
            return ClassificationResult(
                is_chain=True,
                is_compounding=False,
                confidence=1.0,
                explanation=f"Matched chain keyword: {kw}",
                source=ClassificationSource.RULE_BASED
            )

    # 2. Compounding pharmacies - only match whole words to avoid false positives like "Non-Compounding"
    if _token_match(name, "compounding"):
        return ClassificationResult(
            is_chain=False,
            is_compounding=True,
            confidence=0.95,
            explanation="Compounding pharmacy keyword detected",
            source=ClassificationSource.RULE_BASED
        )

    # 3. Default independent
    return ClassificationResult(
        is_chain=False,
        is_compounding=False,
        confidence=0.5,
        explanation="No chain identifiers found",
        source=ClassificationSource.RULE_BASED
    )


def query_perplexity(pharmacy: Union[Dict, PharmacyData]) -> ClassificationResult:
    """Stub LLM call. Tests usually monkey-patch this function.
    
    Args:
        pharmacy: Pharmacy data as either a dictionary or PharmacyData instance
        
    Returns:
        ClassificationResult with LLM-based classification
    """
    logger.debug("query_perplexity stub called; returning fixed result")
    return ClassificationResult(
        is_chain=False,
        is_compounding=False,
        confidence=0.75,
        explanation="Stub LLM result",
        source=ClassificationSource.PERPLEXITY
    )


# Module-level cache for storing classification results
from typing import Dict, Any, Union

_classification_cache: Dict[str, ClassificationResult] = {}

def _get_cache_key(pharmacy: Union[Dict, PharmacyData, None], use_llm: bool = True) -> str:
    """Generate a consistent cache key for a pharmacy.
    
    This function generates the same cache key for equivalent pharmacy data,
    regardless of whether it's provided as a dictionary or PharmacyData object.
    
    Args:
        pharmacy: Pharmacy data as either a dictionary, PharmacyData instance, or None
        use_llm: Whether to use LLM for classification (affects cache key)
        
    Returns:
        String key for caching in the format "normalized_name:normalized_address:use_llm"
        
    Raises:
        ValueError: If pharmacy is None
    """
    if pharmacy is None:
        raise ValueError("Pharmacy data cannot be None")
# Class interface
###############################################################################

class Classifier:
    """Classifies pharmacies using a combination of rule-based and LLM-based approaches.
    
    This class provides a clean interface for classifying pharmacies, with optional
    fallback to a PerplexityClient for LLM-based classification when rule-based
    classification has low confidence.
    """

    def __init__(self, client: Optional["PerplexityClient"] = None):
        """Initialize the classifier with an optional PerplexityClient."""
        from .perplexity_client import PerplexityClient
        self._client = client or PerplexityClient()

    def classify_pharmacy(
        self, 
        pharmacy: Union[Dict, PharmacyData],
        use_llm: bool = True
    ) -> ClassificationResult:
        """Classify a single pharmacy.

        Args:
            pharmacy: Pharmacy data as either a dictionary or PharmacyData instance.
                    Empty dictionaries are allowed and will be processed normally.
            use_llm: Whether to use LLM for classification when rule-based has low confidence
            
        Returns:
            ClassificationResult with the classification results
            
        Raises:
            ValueError: If pharmacy is None
        """
        if pharmacy is None:
            raise ValueError("Pharmacy data cannot be None")
            
        # Check cache first
        cache_key = _get_cache_key(pharmacy)
        if cache_key in _classification_cache:
            logger.debug("Cache hit for pharmacy: %s", cache_key)
            cached_result = _classification_cache[cache_key]
            # Return a copy with the method updated to CACHED
            return ClassificationResult(
                is_chain=cached_result.is_chain,
                is_compounding=cached_result.is_compounding,
                confidence=cached_result.confidence,
                explanation=f"Cached: {cached_result.explanation}",
                source=ClassificationSource.CACHE,
                pharmacy_data=cached_result.pharmacy_data,
                model=cached_result.model
            )
            
        # First try rule-based classification
        rule_result = rule_based_classify(pharmacy)
        
        # If we have high confidence or found a compounding pharmacy, cache and return
        if rule_result.confidence >= 0.9 or rule_result.is_compounding:
            _classification_cache[cache_key] = rule_result
            return rule_result
            
        # If LLM is disabled or no client available, cache and return rule-based result
        if not use_llm or self._client is None:
            _classification_cache[cache_key] = rule_result
            return rule_result
            
        # Try LLM classification
        try:
            llm_result = self._client.classify_pharmacy(pharmacy)

            # Cache and return the result with higher confidence
            final_result = llm_result if llm_result.confidence >= rule_result.confidence else rule_result
            _classification_cache[cache_key] = final_result
            return final_result

        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            # Cache the rule-based result on LLM failure
            _classification_cache[cache_key] = rule_result
            return rule_result
            
    def batch_classify_pharmacies(
        self,
        pharmacies: List[Union[Dict, PharmacyData]],
        use_llm: bool = True
    ) -> List[Optional[ClassificationResult]]:
        """Classify multiple pharmacies in batch mode.
        
        Args:
            pharmacies: List of pharmacy data as either dictionaries or PharmacyData instances
            use_llm: Whether to use LLM for classification when rule-based has low confidence
            
        Returns:
            List of ClassificationResult objects, one for each pharmacy.
            Failed classifications will have None in the corresponding position.
        """
        if not pharmacies:
            return []
            
        results: List[Optional[ClassificationResult]] = []
        
        # Process each pharmacy, handling exceptions individually
        for pharmacy in pharmacies:
            try:
                result = self.classify_pharmacy(pharmacy, use_llm=use_llm)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to classify pharmacy: {e}")
                results.append(None)
                
        return results


###############################################################################
# Re-export list
###############################################################################

__all__ = [
    "CHAIN_IDENTIFIERS",
    "rule_based_classify",
    "Classifier",
]