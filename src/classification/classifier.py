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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the Perplexity client
from .perplexity_client import classify_pharmacy as perplexity_classify
from .cache import Cache, get_cache

# Known chain pharmacy identifiers
CHAIN_IDENTIFIERS = [
    'CVS', 'Walgreens', 'Rite Aid', 'Walmart', 'Target', 'Walgreens',
    'Rite Aid', 'Costco', 'Kroger', 'Albertsons', 'Publix', 'Safeway',
    'Wegmans', 'Giant', 'Stop & Shop', 'H-E-B', 'Meijer', 'Hy-Vee',
    'Publix', 'Kaiser Permanente', 'Kroger', 'Rite Aid', 'Walgreens'
]

def query_perplexity(pharmacy_data: Dict[str, Any], cache: Optional[Cache] = None) -> Dict[str, Any]:
    """
    Query the Perplexity API to classify a pharmacy with caching support.
    
    Args:
        pharmacy_data: Dictionary containing pharmacy information
        cache: Optional cache instance to use for storing/retrieving results
        
    Returns:
        Dict with classification results
    """
    # Create a cache key based on the pharmacy data
    cache_key = f"pharmacy_{hash(frozenset(sorted(pharmacy_data.items())))}"
    
    # Try to get from cache first
    if cache is not None:
        cached_result = cache.get(cache_key)
        if cached_result is not None:
            logger.debug(f"Cache hit for pharmacy: {pharmacy_data.get('name', 'unknown')}")
            return cached_result
    
    try:
        # Call the actual Perplexity API
        result = perplexity_classify(pharmacy_data)
        
        # Add method info
        result["method"] = "llm"
        
        # Cache the result
        if cache is not None:
            cache.set(cache_key, result)
            
        return result
        
    except Exception as e:
        logger.error(f"Error querying Perplexity API: {e}")
        return {
            "is_chain": False,
            "confidence": 0.0,
            "reason": f"Error: {str(e)}",
            "method": "error"
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
    confidence_threshold: float = 0.8,
    cache_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Classify a pharmacy as independent or chain with caching support.
    
    Args:
        pharmacy_data: Dictionary containing pharmacy information
        confidence_threshold: Minimum confidence required for LLM classification
        cache_dir: Optional directory for cache storage
        
    Returns:
        Dict with classification results
    """
    # Create a cache instance if a directory is provided
    cache = None
    if cache_dir:
        cache = get_cache("pharmacy_classification", cache_dir=cache_dir)
    
    # First try LLM classification
    try:
        llm_result = query_perplexity(pharmacy_data, cache=cache)
        
        # If confidence is high enough, return LLM result
        if llm_result.get('confidence', 0) >= confidence_threshold:
            return llm_result
            
        # Otherwise fall back to rule-based
        rule_result = rule_based_classify(pharmacy_data)
        return rule_result
        
    except Exception as e:
        logger.error(f"Error in classification: {e}")
        # Fall back to rule-based on error
        return rule_based_classify(pharmacy_data)

def batch_classify_pharmacies(
    pharmacies: pd.DataFrame,
    confidence_threshold: float = 0.8,
    batch_size: int = 10,
    cache_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Classify multiple pharmacies in batches with caching support.
    
    Args:
        pharmacies: DataFrame containing pharmacy data
        confidence_threshold: Minimum confidence required for LLM classification
        batch_size: Number of pharmacies to process in each batch
        cache_dir: Optional directory for cache storage
        
    Returns:
        DataFrame with added classification columns
    """
    results = []
    
    # Create a cache instance if a directory is provided
    cache = None
    if cache_dir:
        cache = get_cache("pharmacy_classification", cache_dir=cache_dir)
    
    # Process pharmacies in batches
    for i in range(0, len(pharmacies), batch_size):
        batch = pharmacies.iloc[i:i + batch_size].to_dict('records')
        
        for pharmacy_data in batch:
            # Classify the pharmacy
            classification = classify_pharmacy(
                pharmacy_data, 
                confidence_threshold=confidence_threshold,
                cache_dir=cache_dir
            )
            
            # Add classification results to the row
            row_dict = pharmacy_data.copy()
            row_dict.update({
                'is_chain': classification['is_chain'],
                'confidence': classification.get('confidence', 0.0),
                'classification_method': classification.get('method', 'unknown'),
                'classification_reason': classification.get('reason', '')
            })
            results.append(row_dict)
    
    return pd.DataFrame(results)

# For backward compatibility
classify = classify_pharmacy
