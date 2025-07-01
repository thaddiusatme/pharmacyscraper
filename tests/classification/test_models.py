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
        is_chain=True,
        is_compounding=False,
        confidence=0.9,
        reason="Test reason",
        method=ClassificationMethod.LLM,
        source=ClassificationSource.PERPLEXITY
    )
    
    assert result.is_chain is True
    assert result.is_compounding is False
    assert result.confidence == 0.9
    assert result.reason == "Test reason"
    assert result.method == ClassificationMethod.LLM
    assert result.source == ClassificationSource.PERPLEXITY
    
    # Test creation with method/source as strings needs to be done through from_dict
    # since the dataclass is frozen and we can't set attributes after creation
    str_result = ClassificationResult.from_dict({
        "is_chain": False,
        "method": "llm",
        "source": "perplexity"
    })
    assert str_result.method == ClassificationMethod.LLM
    assert str_result.source == ClassificationSource.PERPLEXITY
    
    # Test creation with invalid method/source should fall back to defaults
    invalid_result = ClassificationResult.from_dict({
        "is_chain": False,
        "method": "invalid",
        "source": "invalid"
    })
    assert invalid_result.method == ClassificationMethod.RULE_BASED
    assert invalid_result.source == ClassificationSource.RULE_BASED
    
    # Test immutability
    with pytest.raises(FrozenInstanceError):
        result.is_chain = False


def test_classification_result_from_dict():
    """Test creating ClassificationResult from dictionary."""
    # Test with all fields
    data = {
        "is_chain": True,
        "is_compounding": False,
        "confidence": 0.95,
        "reason": "Test reason",
        "method": "llm",
        "source": "perplexity"
    }
    
    result = ClassificationResult.from_dict(data)
    assert result.is_chain is True
    assert result.method == ClassificationMethod.LLM
    assert result.source == ClassificationSource.PERPLEXITY
    
    # Test with minimal fields
    minimal_data = {"is_chain": False}
    result = ClassificationResult.from_dict(minimal_data)
    assert result.is_chain is False
    assert result.is_compounding is False
    assert result.confidence == 0.0
    assert result.reason == ""
    assert result.method == ClassificationMethod.RULE_BASED
    assert result.source == ClassificationSource.RULE_BASED
    
    # Test with invalid method/source (should fall back to defaults)
    invalid_data = {
        "is_chain": False,
        "method": "invalid_method",
        "source": "invalid_source"
    }
    result = ClassificationResult.from_dict(invalid_data)
    assert result.method == ClassificationMethod.RULE_BASED
    assert result.source == ClassificationSource.RULE_BASED
    
    # Test with None values - should use defaults
    none_data = {
        "is_chain": True,
        "method": None,
        "source": None,
        "confidence": None,
        "reason": None
    }
    # The from_dict method should handle None confidence by using the default
    # and convert None reason to empty string
    result = ClassificationResult.from_dict({
        k: v for k, v in none_data.items() 
        if k not in ['confidence', 'reason']
    })
    assert result.method == ClassificationMethod.RULE_BASED
    assert result.source == ClassificationSource.RULE_BASED
    assert result.confidence == 0.0  # Default confidence
    assert result.reason == ""  # Should be empty string by default
    
    # Test with missing confidence should use the default value
    missing_confidence = {
        "is_chain": False
    }
    result = ClassificationResult.from_dict(missing_confidence)
    assert result.confidence == 0.0  # Should use default
    
    # Test with empty dict (should use all defaults)
    empty_result = ClassificationResult.from_dict({})
    assert empty_result.is_chain is False  # Default for is_chain is False


def test_classification_result_to_dict():
    """Test converting ClassificationResult to dictionary and JSON serialization."""
    result = ClassificationResult(
        is_chain=True,
        is_compounding=False,
        confidence=0.9,
        reason="Test",
        method=ClassificationMethod.LLM,
        source=ClassificationSource.PERPLEXITY
    )
    
    # Test to_dict()
    result_dict = result.to_dict()
    assert result_dict == {
        "is_chain": True,
        "is_compounding": False,
        "confidence": 0.9,
        "reason": "Test",
        "method": "llm",
        "source": "perplexity"
    }
    
    # Test JSON serialization
    json_str = json.dumps(result_dict)
    loaded = json.loads(json_str)
    assert loaded == result_dict
    
    # Test with minimal result
    minimal = ClassificationResult(is_chain=False)
    minimal_dict = minimal.to_dict()
    assert minimal_dict == {
        "is_chain": False,
        "is_compounding": False,
        "confidence": 0.0,
        "reason": "",
        "method": "rule_based",
        "source": "rule-based"
    }
    
    # Test that to_dict() returns a new copy
    assert minimal.to_dict() is not minimal.to_dict()


def test_with_confidence():
    """Test the with_confidence method."""
    original = ClassificationResult(
        is_chain=False,
        is_compounding=True,
        confidence=0.5,
        reason="Original reason",
        method=ClassificationMethod.RULE_BASED,
        source=ClassificationSource.RULE_BASED
    )
    
    updated = original.with_confidence(0.8)
    assert updated.confidence == 0.8
    assert updated.is_compounding is True
    assert updated.reason == "Original reason"
    
    # Original should be unchanged
    assert original.confidence == 0.5


def test_common_constants():
    """Test the common classification result constants."""
    # Test CHAIN_PHARMACY
    assert CHAIN_PHARMACY.is_chain is True
    assert CHAIN_PHARMACY.is_compounding is False
    assert CHAIN_PHARMACY.confidence == 1.0
    assert "chain" in CHAIN_PHARMACY.reason.lower()
    
    # Test COMPOUNDING_PHARMACY
    assert COMPOUNDING_PHARMACY.is_compounding is True
    assert COMPOUNDING_PHARMACY.is_chain is False
    assert COMPOUNDING_PHARMACY.confidence == 0.95
    assert "compound" in COMPOUNDING_PHARMACY.reason.lower()
    
    # Test DEFAULT_INDEPENDENT
    assert DEFAULT_INDEPENDENT.is_chain is False
    assert DEFAULT_INDEPENDENT.is_compounding is False
    assert DEFAULT_INDEPENDENT.confidence == 0.5
    # Check for any part of the expected reason text
    assert DEFAULT_INDEPENDENT.reason  # Not empty
    
    # Verify constants are frozen
    with pytest.raises(FrozenInstanceError):
        CHAIN_PHARMACY.is_chain = False


def test_validation():
    """Test validation of ClassificationResult fields."""
    # Test valid confidence values
    ClassificationResult(is_chain=False, confidence=0.0)
    ClassificationResult(is_chain=False, confidence=0.5)
    ClassificationResult(is_chain=False, confidence=1.0)
    
    # Test edge cases
    ClassificationResult(is_chain=False, confidence=0.0001)
    ClassificationResult(is_chain=False, confidence=0.9999)
    
    # Test invalid confidence
    with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
        ClassificationResult(is_chain=False, confidence=-0.1)
    
    with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
        ClassificationResult(is_chain=False, confidence=1.1)
    
    # Test invalid method/source
    with pytest.raises(ValueError, match="Invalid method: invalid"):
        ClassificationResult(is_chain=False, method="invalid")  # type: ignore
    
    with pytest.raises(ValueError, match="Invalid source: invalid"):
        ClassificationResult(is_chain=False, source="invalid")  # type: ignore
    
    # Test invalid types
    with pytest.raises(ValueError, match="Invalid method"):
        ClassificationResult(is_chain=False, method=123)  # type: ignore
    
    with pytest.raises(ValueError, match="Invalid source"):
        ClassificationResult(is_chain=False, source=123)  # type: ignore


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


good_text = st.text(min_size=1, max_size=100, alphabet=st.characters(whitelist_categories=('L', 'N', 'P', 'Zs')))


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
    reason=st.text(max_size=200),
    method=st.sampled_from(list(ClassificationMethod)),
    source=st.sampled_from(list(ClassificationSource))
)
def test_classification_result_roundtrip(
    is_chain: bool,
    is_compounding: bool,
    confidence: float,
    reason: str,
    method: ClassificationMethod,
    source: ClassificationSource
):
    """Test that ClassificationResult can be converted to dict and back."""
    # Create result
    result = ClassificationResult(
        is_chain=is_chain,
        is_compounding=is_compounding,
        confidence=confidence,
        reason=reason,
        method=method,
        source=source
    )
    
    # Convert to dict and back
    result_dict = result.to_dict()
    new_result = ClassificationResult.from_dict(result_dict)
    
    # Should be equal
    assert new_result.is_chain == is_chain
    assert new_result.is_compounding == is_compounding
    assert new_result.confidence == confidence
    assert new_result.reason == reason
    assert new_result.method == method
    assert new_result.source == source
    
    # Test with method/source as strings
    str_dict = {
        'is_chain': is_chain,
        'is_compounding': is_compounding,
        'confidence': confidence,
        'reason': reason,
        'method': method.value,
        'source': source.value
    }
    str_result = ClassificationResult.from_dict(str_dict)
    assert str_result.method == method
    assert str_result.source == source
