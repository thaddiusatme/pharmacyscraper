"""
Pharmacy classification module.

This module provides functionality to classify pharmacies as independent or chain
using both LLM-based and rule-based approaches.
"""

__version__ = "0.1.0"


from .classifier import (
    Classifier,
    rule_based_classify,
    CHAIN_IDENTIFIERS
)
from .perplexity_client import (
    PerplexityClient,
    RateLimitError,
    InvalidRequestError,
    ResponseParseError,
)

__all__ = [
    "Classifier",
    "PerplexityClient",
    "rule_based_classify",
    "CHAIN_IDENTIFIERS",
    "RateLimitError",
    "InvalidRequestError",
    "ResponseParseError",
]
