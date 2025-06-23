"""
Top-level package for the Pharmacy Scrape project.

Independent Pharmacy Verification Project

This package provides functionality to classify and verify independent pharmacies.
"""

__version__ = "0.1.0"

from .classification import (
    Classifier,
    PerplexityClient
)

__all__ = [
    "Classifier",
    "PerplexityClient",
]
