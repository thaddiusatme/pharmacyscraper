"""
Top-level package for the Pharmacy Scrape project.

Independent Pharmacy Verification Project

This package provides functionality to classify and verify independent pharmacies.
"""

__version__ = "0.1.0"

from .classification.classifier import (
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
