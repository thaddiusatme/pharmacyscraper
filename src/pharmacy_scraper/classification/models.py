"""
Data models for pharmacy classification.

This module defines the core data structures used throughout the classification system.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Optional, Any, List, Union, Literal, Callable


class ClassificationMethod(str, Enum):
    """The method used to classify a pharmacy."""
    RULE_BASED = "rule_based"
    LLM = "llm"
    CACHED = "cached"


class ClassificationSource(str, Enum):
    """The source of the classification result."""
    RULE_BASED = "rule-based"
    PERPLEXITY = "perplexity"
    CACHE = "cache"


# Type alias for confidence values (0.0 to 1.0)
Confidence = float


@dataclass(frozen=True)
class PharmacyData:
    """Structured representation of pharmacy data for classification.
    
    Attributes:
        name: The name of the pharmacy.
        address: The full address of the pharmacy.
        phone: The phone number of the pharmacy.
        categories: Categories or types of the pharmacy.
        website: The website URL of the pharmacy.
        raw_data: Original raw data dictionary for backward compatibility.
    """
    name: str
    address: Optional[str] = None
    phone: Optional[str] = None
    categories: Optional[str] = None
    website: Optional[str] = None
    raw_data: Dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PharmacyData:
        """Create a PharmacyData instance from a dictionary.
        
        Args:
            data: Dictionary containing pharmacy data with optional keys:
                - name or title: Pharmacy name
                - address: Full address
                - phone: Phone number
                - categories: Categories or types
                - website: Website URL
                - Any other fields will be stored in raw_data
                
        Returns:
            A new PharmacyData instance.
        """
        # Extract known fields and store the rest in raw_data
        known_fields = {
            'name': data.get('name') or data.get('title', ''),
            'address': data.get('address'),
            'phone': data.get('phone'),
            'categories': data.get('categories') or data.get('categoryName'),
            'website': data.get('website'),
        }
        
        # Create a copy of the input data without the known fields
        raw_data = data.copy()
        for field_name in ['name', 'title', 'address', 'phone', 'categories', 'categoryName', 'website']:
            raw_data.pop(field_name, None)
            
        return cls(**known_fields, raw_data=raw_data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the PharmacyData instance to a dictionary.
        
        Returns:
            A dictionary representation of the pharmacy data.
        """
        result = {
            'name': self.name,
            'address': self.address,
            'phone': self.phone,
            'categories': self.categories,
            'website': self.website,
            **self.raw_data
        }
        return {k: v for k, v in result.items() if v is not None}


@dataclass(frozen=True)
class ClassificationResult:
    """Represents the result of a pharmacy classification.

    Attributes:
        classification: The primary classification (e.g., 'chain', 'independent').
        is_chain: Boolean indicating if the pharmacy is a chain.
        is_compounding: Boolean indicating if the pharmacy does compounding.
        confidence: The confidence score of the classification (0.0 to 1.0).
        explanation: The explanation for the classification.
        source: The source of the classification (e.g., 'llm', 'rule-based').
        model: The model used for classification, if applicable.
        pharmacy_data: The input data used for classification.
        error: An error message, if any occurred.
    """
    classification: Optional[str] = None
    is_chain: Optional[bool] = None
    is_compounding: Optional[bool] = None
    confidence: Optional[Confidence] = None
    explanation: Optional[str] = None
    source: Optional[ClassificationSource] = None
    model: Optional[str] = None
    pharmacy_data: Optional[PharmacyData] = None
    error: Optional[str] = None

    @property
    def cached(self) -> bool:
        """Returns True if the result was retrieved from the cache."""
        return self.source == ClassificationSource.CACHE

    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary."""
        data = {
            'classification': self.classification,
            'is_chain': self.is_chain,
            'is_compounding': self.is_compounding,
            'confidence': self.confidence,
            'explanation': self.explanation,
            'source': self.source.value if self.source else None,
            'model': self.model,
            'cached': self.cached,  # Use the property here
            'error': self.error,
            'pharmacy_data': self.pharmacy_data.to_dict() if self.pharmacy_data else None
        }
        return {k: v for k, v in data.items() if v is not None}


# Common result constants for reuse
CHAIN_PHARMACY = ClassificationResult(
    classification="chain",
    is_chain=True,
    is_compounding=False,
    confidence=1.0,
    explanation="Chain pharmacy identified",
    source=ClassificationSource.RULE_BASED,
    model=None,
    pharmacy_data=None,
    error=None
)

COMPOUNDING_PHARMACY = ClassificationResult(
    classification="independent",
    is_chain=False,
    is_compounding=True,
    confidence=0.95,
    explanation="Compounding pharmacy identified",
    source=ClassificationSource.RULE_BASED,
    model=None,
    pharmacy_data=None,
    error=None
)

DEFAULT_INDEPENDENT = ClassificationResult(
    classification="independent",
    is_chain=False,
    is_compounding=False,
    confidence=0.5,
    explanation="No chain identifiers found",
    source=ClassificationSource.RULE_BASED,
    model=None,
    pharmacy_data=None,
    error=None
)

# Type alias for classification functions
ClassifierFunction = Callable[[PharmacyData], ClassificationResult]
