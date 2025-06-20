"""
Pharmacy classification module.

This module provides functionality to classify pharmacies as independent or chain
using both LLM-based and rule-based approaches.
"""

__version__ = "0.1.0"

from .classifier import (
    classify_pharmacy,
    rule_based_classify,
    batch_classify_pharmacies,
    classify  # For backward compatibility
)

__all__ = [
    'classify_pharmacy',
    'rule_based_classify',
    'batch_classify_pharmacies',
    'classify'
]
