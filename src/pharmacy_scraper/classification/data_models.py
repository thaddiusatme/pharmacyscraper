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
    """Structured representation of a pharmacy classification result.
    
    Attributes:
        is_chain: Whether the pharmacy is part of a chain.
        is_compounding: Whether the pharmacy does compounding.
        confidence: Confidence score between 0.0 and 1.0.
        reason: Human-readable explanation of the classification.
        method: The classification method used.
        source: The source of the classification.
    """
    is_chain: bool
    is_compounding: bool = False
    confidence: Confidence = 0.0
    reason: str = ""
    method: ClassificationMethod = ClassificationMethod.RULE_BASED
    source: ClassificationSource = ClassificationSource.RULE_BASED
    
    def __post_init__(self) -> None:
        """Validate the classification result fields."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")
            
        if not isinstance(self.method, ClassificationMethod):
            try:
                self.method = ClassificationMethod(self.method)
            except ValueError as e:
                raise ValueError(f"Invalid method: {self.method}") from e
                
        if not isinstance(self.source, ClassificationSource):
            try:
                self.source = ClassificationSource(self.source)
            except ValueError as e:
                raise ValueError(f"Invalid source: {self.source}") from e
                
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ClassificationResult:
        """Create a ClassificationResult from a dictionary.
        
        Args:
            data: Dictionary containing classification result data.
                Expected keys: is_chain, is_compounding, confidence, reason,
                method, source.
                
        Returns:
            A new ClassificationResult instance.
            
        Raises:
            ValueError: If required fields are missing or invalid.
        """
        # Handle method and source conversion from strings if needed
        method = data.get('method')
        if isinstance(method, str):
            try:
                method = ClassificationMethod(method)
            except ValueError:
                method = ClassificationMethod.RULE_BASED
        
        source = data.get('source')
        if isinstance(source, str):
            try:
                source = ClassificationSource(source)
            except ValueError:
                source = ClassificationSource.RULE_BASED
        
        return cls(
            is_chain=bool(data.get('is_chain', False)),
            is_compounding=bool(data.get('is_compounding', False)),
            confidence=float(data.get('confidence', 0.0)),
            reason=str(data.get('reason', '')),
            method=method or ClassificationMethod.RULE_BASED,
            source=source or ClassificationSource.RULE_BASED
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the ClassificationResult to a dictionary.
        
        Returns:
            A dictionary representation of the classification result.
        """
        return {
            'is_chain': self.is_chain,
            'is_compounding': self.is_compounding,
            'confidence': self.confidence,
            'reason': self.reason,
            'method': self.method.value,
            'source': self.source.value
        }
    
    def with_confidence(self, confidence: Confidence) -> ClassificationResult:
        """Return a new instance with updated confidence.
        
        Args:
            confidence: New confidence value (0.0 to 1.0).
            
        Returns:
            A new ClassificationResult with updated confidence.
        """
        return ClassificationResult(
            is_chain=self.is_chain,
            is_compounding=self.is_compounding,
            confidence=confidence,
            reason=self.reason,
            method=self.method,
            source=self.source
        )


# Common result constants for reuse
CHAIN_PHARMACY = ClassificationResult(
    is_chain=True,
    is_compounding=False,
    confidence=1.0,
    reason="Chain pharmacy identified",
    method=ClassificationMethod.RULE_BASED,
    source=ClassificationSource.RULE_BASED
)

COMPOUNDING_PHARMACY = ClassificationResult(
    is_chain=False,
    is_compounding=True,
    confidence=0.95,
    reason="Compounding pharmacy identified",
    method=ClassificationMethod.RULE_BASED,
    source=ClassificationSource.RULE_BASED
)

DEFAULT_INDEPENDENT = ClassificationResult(
    is_chain=False,
    is_compounding=False,
    confidence=0.5,
    reason="No chain identifiers found",
    method=ClassificationMethod.RULE_BASED,
    source=ClassificationSource.RULE_BASED
)

# Type alias for classification functions
ClassifierFunction = Callable[[PharmacyData], ClassificationResult]
