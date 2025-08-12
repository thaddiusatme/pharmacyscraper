"""Tests for pharmacy classification data models."""
import pytest
import json
from dataclasses import FrozenInstanceError, asdict
from typing import Dict, Any, Optional
from hypothesis import given, strategies as st

from pharmacy_scraper.classification.models import (
    ClassificationMethod,
    ClassificationSource,
    Confidence,
    PharmacyData,
    ClassificationResult,
    CHAIN_PHARMACY,
    COMPOUNDING_PHARMACY,
    DEFAULT_INDEPENDENT
)


def test_pharmacy_data_creation():
    """Test basic PharmacyData creation and properties."""
    data = {
        "name": "Test Pharmacy",
        "address": "123 Main St",
        "phone": "555-1234",
        "categories": "Pharmacy, Compounding",
        "website": "https://testpharmacy.com",
        "extra_field": "extra value",
        "categoryName": "Pharmacy"  # Test alternative field name
    }
    
    # Test creation with all fields
    pharmacy = PharmacyData.from_dict(data)
    assert pharmacy.name == "Test Pharmacy"
    assert pharmacy.address == "123 Main St"
    assert pharmacy.phone == "555-1234"
    assert pharmacy.categories == "Pharmacy, Compounding"  # Should prefer 'categories' over 'categoryName'
    assert pharmacy.website == "https://testpharmacy.com"
    assert pharmacy.raw_data == {"extra_field": "extra value"}
    
    # Test to_dict
    result = pharmacy.to_dict()
    assert result["name"] == "Test Pharmacy"
    assert result["categories"] == "Pharmacy, Compounding"
    assert "categoryName" not in result  # Should not include alternative field names
    assert result["extra_field"] == "extra value"
    
    # Test direct instantiation
    direct = PharmacyData(
        name="Direct Pharmacy",
        address="456 Oak St",
        raw_data={"test": "value"}
    )
    assert direct.name == "Direct Pharmacy"
    assert direct.raw_data == {"test": "value"}
    
    # Test immutability
    with pytest.raises(FrozenInstanceError):
        pharmacy.name = "New Name"


def test_pharmacy_data_optional_fields():
    """Test PharmacyData with optional fields and edge cases."""
    # Test with minimal required fields
    pharmacy = PharmacyData.from_dict({"name": "Minimal Pharmacy"})
    assert pharmacy.name == "Minimal Pharmacy"
    assert pharmacy.address is None
    assert pharmacy.phone is None
    assert pharmacy.categories is None
    assert pharmacy.website is None
    assert pharmacy.raw_data == {}
    
    # Test with title instead of name
    pharmacy = PharmacyData.from_dict({"title": "Pharmacy with Title"})
    assert pharmacy.name == "Pharmacy with Title"
    
    # Test with empty strings and whitespace
    pharmacy = PharmacyData.from_dict({
        "name": "  ",
        "address": "",
        "phone": "  ",
        "categories": "",
        "website": ""
    })
    assert pharmacy.name == "  "  # Empty strings are preserved
    assert pharmacy.address == ""
    assert pharmacy.phone == "  "
    # Categories should be None when empty string is provided
    assert pharmacy.categories is None
    assert pharmacy.website == ""
    
    # Test with None values in dict
    pharmacy = PharmacyData.from_dict({
        "name": None,
        "address": None,
        "phone": None,
        "categories": None,
        "website": None
    })
    assert pharmacy.name == ""  # None is converted to empty string for name
    assert pharmacy.address is None
    assert pharmacy.phone is None
    assert pharmacy.categories is None
    assert pharmacy.website is None
    
    # Test with raw_data containing None values
    pharmacy = PharmacyData.from_dict({
        "name": "Test",
        "extra": None,
        "nested": {"key": None}
    })
    assert pharmacy.raw_data == {"extra": None, "nested": {"key": None}}


def test_classification_result_creation():
    """Test basic ClassificationResult creation and properties."""
    # Test creation with all fields
    result = ClassificationResult(
        classification="chain",
        is_chain=True,
        is_compounding=False,
        confidence=0.9,
        explanation="Test reason",
        source=ClassificationSource.PERPLEXITY
    )
    
    assert result.is_chain is True
    assert result.is_compounding is False
    assert result.confidence == 0.9
    assert result.explanation == "Test reason"
    assert result.source == ClassificationSource.PERPLEXITY
    
    # Test creation with minimal fields
    min_result = ClassificationResult(
        is_chain=False,
        source=ClassificationSource.PERPLEXITY
    )
    assert min_result.source == ClassificationSource.PERPLEXITY
    
    # Test immutability
    with pytest.raises(FrozenInstanceError):
        result.is_chain = False


def test_classification_result_to_dict():
    """Test conversion of ClassificationResult to dictionary."""
    # Create a ClassificationResult
    result = ClassificationResult(
        classification="chain",
        is_chain=True,
        is_compounding=False,
        confidence=0.9,
        explanation="Test reason",
        source=ClassificationSource.PERPLEXITY,
        model="test-model"
    )
    
    # Convert to dictionary
    data = result.to_dict()
    
    # Verify dictionary contents
    assert data["is_chain"] is True
    assert data["is_compounding"] is False
    assert data["confidence"] == 0.9
    assert data["explanation"] == "Test reason"
    assert data["source"] == "perplexity"
    assert data["model"] == "test-model"
    
    # Test JSON serialization
    json_str = json.dumps(data)
    loaded = json.loads(json_str)
    assert loaded == data
    
    # Test minimal result
    minimal = ClassificationResult(is_chain=False)
    minimal_dict = minimal.to_dict()
    assert "is_chain" in minimal_dict
    assert minimal_dict["is_chain"] is False
    assert "cached" in minimal_dict
    assert minimal_dict["cached"] is False
    
    # Test that to_dict() returns a new copy
    assert minimal.to_dict() is not minimal.to_dict()


def test_with_confidence():
    """Test that confidence value is properly handled."""
    # Create ClassificationResults with different confidence values
    high_confidence = ClassificationResult(
        classification="chain",
        is_chain=True,
        confidence=0.95,
        explanation="High confidence",
        source=ClassificationSource.PERPLEXITY
    )
    
    low_confidence = ClassificationResult(
        classification="independent",
        is_chain=False,
        confidence=0.3,
        explanation="Low confidence",
        source=ClassificationSource.PERPLEXITY
    )
    
    assert high_confidence.confidence == 0.95
    assert low_confidence.confidence == 0.3


def test_common_constants():
    """Test the common constant ClassificationResult instances."""
    # Check CHAIN_PHARMACY constant
    assert CHAIN_PHARMACY.classification == "chain"
    assert CHAIN_PHARMACY.is_chain is True
    assert CHAIN_PHARMACY.is_compounding is False
    assert CHAIN_PHARMACY.confidence == 1.0
    assert "chain" in CHAIN_PHARMACY.explanation.lower()
    assert CHAIN_PHARMACY.source == ClassificationSource.RULE_BASED
    
    # Check COMPOUNDING_PHARMACY constant
    assert COMPOUNDING_PHARMACY.classification == "independent"
    assert COMPOUNDING_PHARMACY.is_chain is False
    assert COMPOUNDING_PHARMACY.is_compounding is True
    assert COMPOUNDING_PHARMACY.confidence == 0.95
    assert "compound" in COMPOUNDING_PHARMACY.explanation.lower()
    
    # Check DEFAULT_INDEPENDENT constant
    assert DEFAULT_INDEPENDENT.classification == "independent"
    assert DEFAULT_INDEPENDENT.is_chain is False
    assert DEFAULT_INDEPENDENT.is_compounding is False
    # Check for any part of the expected explanation text
    assert DEFAULT_INDEPENDENT.explanation  # Not empty


def test_cached_property():
    """Test the cached property of ClassificationResult."""
    # Create a result with cache source
    cached_result = ClassificationResult(
        classification="chain",
        is_chain=True,
        confidence=0.9,
        source=ClassificationSource.CACHE
    )
    
    # Create a result with non-cache source
    non_cached_result = ClassificationResult(
        classification="chain",
        is_chain=True,
        confidence=0.9,
        source=ClassificationSource.PERPLEXITY
    )
    
    assert cached_result.cached is True
    assert non_cached_result.cached is False


def test_classification_result_with_none_values():
    """Test ClassificationResult with None values."""
    # Create a result with some None values
    result = ClassificationResult(
        classification=None,
        is_chain=None,
        is_compounding=None,
        confidence=None,
        explanation=None,
        source=None,
        model=None,
        pharmacy_data=None,
        error=None
    )
    
    # Verify defaults when None is provided
    assert result.classification is None
    assert result.is_chain is None
    assert result.is_compounding is None
    assert result.confidence is None
    assert result.explanation is None
    assert result.source is None
    assert result.model is None
    assert result.pharmacy_data is None
    assert result.error is None


def test_classification_result_immutable():
    """Test that ClassificationResult is immutable (frozen)."""
    result = ClassificationResult(
        classification="independent",
        is_chain=False,
        confidence=0.8,
        explanation="Test explanation"
    )
    
    # Attempting to modify should raise FrozenInstanceError
    with pytest.raises(FrozenInstanceError):
        result.is_chain = True
        
    with pytest.raises(FrozenInstanceError):
        result.confidence = 0.9
        
    with pytest.raises(FrozenInstanceError):
        result.explanation = "New explanation"


# Property-based tests
@st.composite
def pharmacy_data_strategy(draw):
    """Generate random PharmacyData instances for property-based testing."""
    name = draw(st.text(min_size=1, max_size=100))
    address = draw(st.one_of(st.none(), st.text(max_size=200)))
    phone = draw(st.one_of(st.none(), st.text(max_size=20)))
    categories = draw(st.one_of(st.none(), st.text(max_size=200)))
    website = draw(st.one_of(st.none(), st.text(max_size=200)))
    
    # Generate random raw data
    raw_data = {}
    for _ in range(draw(st.integers(0, 5))):
        key = draw(st.text(min_size=1, max_size=20))
        value = draw(st.one_of(
            st.none(),
            st.booleans(),
            st.integers(),
            st.floats(),
            st.text(max_size=50)  # Changed from max_length to max_size
        ))
        # Convert any non-string values to strings to avoid serialization issues
        if value is not None and not isinstance(value, (str, bool, int, float)):
            value = str(value)
        raw_data[key] = value
    
    return PharmacyData(
        name=name,
        address=address,
        phone=phone,
        categories=categories,
        website=website,
        raw_data=raw_data
    )


@given(pharmacy=pharmacy_data_strategy())
def test_pharmacy_data_roundtrip(pharmacy: PharmacyData):
    """Test that PharmacyData can be converted to dict and back."""
    # Skip if name would be empty after stripping
    if not pharmacy.name.strip():
        return
        
    # Convert to dict and back
    data = asdict(pharmacy)
    new_pharmacy = PharmacyData(**data)
    
    # Should be equal
    assert new_pharmacy == pharmacy
    
    # Test to_dict() as well
    dict_data = pharmacy.to_dict()
    assert dict_data["name"] == pharmacy.name
    
    # Test that raw_data is preserved
    assert new_pharmacy.raw_data == pharmacy.raw_data


@given(
    is_chain=st.booleans(),
    is_compounding=st.booleans(),
    confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    explanation=st.text(max_size=200),
    source=st.sampled_from(list(ClassificationSource))
)
def test_classification_result_serialization(
    is_chain: bool,
    is_compounding: bool,
    confidence: float,
    explanation: str,
    source: ClassificationSource
):
    """Test that ClassificationResult serializes consistently."""
    # Create result
    result = ClassificationResult(
        classification="test-classification",
        is_chain=is_chain,
        is_compounding=is_compounding,
        confidence=confidence,
        explanation=explanation,
        source=source
    )
    
    # Convert to dict
    result_dict = result.to_dict()
    
    # Check values match
    assert result_dict["classification"] == "test-classification"
    assert result_dict["is_chain"] == is_chain
    assert result_dict["is_compounding"] == is_compounding
    assert result_dict["confidence"] == confidence
    assert result_dict["explanation"] == explanation
    # Source should be present since classification is not None
    assert "source" in result_dict
    # Defensive check for the flaky test issue
    if source is None:
        # This should never happen with our Hypothesis strategy, but let's be defensive
        assert result_dict["source"] is None
    else:
        assert result_dict["source"] == source.value
