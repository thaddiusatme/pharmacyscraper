"""
Pharmacy classification module.

This module provides functionality to classify pharmacies as independent or chain
using both LLM-based and rule-based approaches.
"""
from typing import Dict, Any, List, Optional
import logging
import pandas as pd
import json
from pathlib import Path
from .cache import cache_wrapper

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the Perplexity client
from .perplexity_client import PerplexityClient, PerplexityAPIError

# Known chain pharmacy identifiers
CHAIN_IDENTIFIERS = [
    'CVS', 'Walgreens', 'Rite Aid', 'Walmart', 'Target', 'Walgreens',
    'Rite Aid', 'Costco', 'Kroger', 'Albertsons', 'Publix', 'Safeway',
    'Wegmans', 'Giant', 'Stop & Shop', 'H-E-B', 'Meijer', 'Hy-Vee',
    'Publix', 'Kaiser Permanente', 'Kroger', 'Rite Aid', 'Walgreens'
]


class Classifier:
    """Classifies pharmacies using a combination of rule-based and LLM approaches."""

    def __init__(self, client: PerplexityClient, rule_based_classifier=None):
        self.client = client
        self.rule_based_classifier = rule_based_classifier or self._default_rule_based_classifier

    @cache_wrapper
    def classify_pharmacy(self, pharmacy_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classifies a single pharmacy. This method is now cached.
        
        Args:
            pharmacy_data: Dictionary containing pharmacy information
            
        Returns:
            Dict with classification results, including source and confidence
        """
        # Handle empty or invalid input
        if not pharmacy_data or not isinstance(pharmacy_data, dict):
            return {
                "is_compounding": False,
                "confidence": 0.0,
                "source": "invalid_input",
                "method": "rule_based"
            }
            
        # First, try the rule-based classifier
        try:
            rule_based_result = self.rule_based_classifier(pharmacy_data)
            if rule_based_result.get('is_compounding'):
                return {**rule_based_result, "method": "rule_based"}
                
            # If rules don't apply, try the LLM classifier
            try:
                llm_result = self.client.classify_pharmacy(pharmacy_data)
                return {**llm_result, "method": "llm"}
            except PerplexityAPIError as e:
                logger.warning(f"LLM classification failed: {e}")
                return {
                    "is_chain": False,
                    "confidence": 0.0,
                    "source": "llm_error",
                    "method": "error"
                }
                
        except Exception as e:
            logger.error(f"Error classifying pharmacy: {e}")
            return {
                "is_compounding": False,
                "confidence": 0.0,
                "source": "error",
                "method": "error"
            }

    def batch_classify_pharmacies(self, pharmacies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Classifies a batch of pharmacies, ensuring each is processed once.
        """
        return [self.classify_pharmacy(pharmacy) for pharmacy in pharmacies]

    def _default_rule_based_classifier(self, pharmacy_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        A simple rule-based classifier to detect compounding pharmacies.
        """
        name = pharmacy_data.get("name", "").lower()
        if "compounding" in name:
            return {"is_compounding": True, "confidence": 1.0, "source": "rule-based"}
        return {"is_compounding": False, "confidence": 0.5, "source": "rule-based"}


def query_perplexity(pharmacy_data: Dict[str, Any], cache_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Query the Perplexity API to classify a pharmacy.

    Args:
        pharmacy_data: Dictionary containing pharmacy information.
        cache_dir: Optional directory for cache storage.

    Returns:
        A dictionary with classification results.
    """
    try:
        # The PerplexityClient now handles its own caching
        client = PerplexityClient(cache_dir=cache_dir)
        result = client.classify_pharmacy(pharmacy_data)

        # Add method info
        result["method"] = "llm"

        return result

    except PerplexityAPIError as e:
        logger.error(f"Error querying Perplexity API: {e}")
        return {
            "is_chain": False,
            "confidence": 0.0,
            "method": "llm_error"
        }


def rule_based_classify(pharmacy_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Classify a pharmacy using rule-based approach.
    
    Args:
        pharmacy_data: Dictionary containing pharmacy information
        
    Returns:
        Dict with classification results
    """
    name = pharmacy_data.get('name', '').lower()
    
    # Check for compounding pharmacies
    if 'compounding' in name:
        return {
            "is_chain": False,
            "confidence": 0.95,
            "reason": "Identified as independent compounding pharmacy",
            "method": "rule_based"
        }
    
    # Check for chain identifiers
    for chain in CHAIN_IDENTIFIERS:
        if chain.lower() in name:
            return {
                "is_chain": True,
                "confidence": 0.99,
                "reason": f"Identified as {chain} chain pharmacy",
                "method": "rule_based"
            }
    
    # Default to independent if no chain indicators found
    return {
        "is_chain": False,
        "confidence": 0.7,
        "reason": "No chain identifiers found, assuming independent",
        "method": "rule_based"
    }


def classify_pharmacy(
    pharmacy_data: Dict[str, Any], 
    use_llm: bool = True,
    use_rules: bool = True,
    cache_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Classifies a pharmacy using a combination of LLM and rule-based methods.
    
    Args:
        pharmacy_data: Dictionary of pharmacy data.
        use_llm: Whether to use the LLM classifier.
        use_rules: Whether to use the rule-based classifier.
        cache_dir: Optional directory for cache storage.

    Returns:
        A dictionary with the final classification.
    """
    # Initialize results
    llm_result = None
    rule_result = None

    # Step 1: Rule-based classification (optional)
    if use_rules:
        rule_result = rule_based_classify(pharmacy_data)
        if rule_result.get('confidence', 0) > 0.7:
            return rule_result

    # Step 2: LLM-based classification (optional)
    if use_llm:
        llm_result = query_perplexity(pharmacy_data, cache_dir=cache_dir)

        # If LLM is confident, use its result
        if llm_result and llm_result.get('confidence', 0) > 0.7:
            return llm_result

    # If both methods are used, return the most confident result
    if llm_result and rule_result:
        if llm_result.get('confidence', 0) > rule_result.get('confidence', 0):
            return llm_result
        else:
            return rule_result

    # If only one method is used, return its result
    if llm_result:
        return llm_result
    if rule_result:
        return rule_result

    # If no method is used, return a default result
    return {
        "is_chain": False,
        "confidence": 0.0,
        "method": "unknown"
    }


def batch_classify_pharmacies(
    pharmacies_data: List[Dict[str, Any]], 
    use_llm: bool = True,
    use_rules: bool = True,
    cache_dir: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Classifies a batch of pharmacies.

    Args:
        pharmacies_data: A list of pharmacy data dictionaries.
        use_llm: Whether to use the LLM classifier.
        use_rules: Whether to use the rule-based classifier.
        cache_dir: Optional directory for cache storage.

    Returns:
        A list of classification results.
    """
    results = []
    for pharmacy_data in pharmacies_data:
        result = classify_pharmacy(
            pharmacy_data,
            use_llm=use_llm,
            use_rules=use_rules,
            cache_dir=cache_dir
        )
        results.append(result)
    return results


# For backward compatibility
classify = classify_pharmacy
