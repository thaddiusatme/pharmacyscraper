"""
Tests for the pharmacy classification module.
"""
import os
import pytest
import logging
import json
import re
from dataclasses import asdict
from pathlib import Path
from dotenv import load_dotenv
from unittest.mock import patch, MagicMock, call
from typing import Dict, Any, List, Optional
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

# Load environment variables from .env file
load_dotenv()

from pharmacy_scraper.classification.classifier import (
    classify_pharmacy,
    batch_classify_pharmacies,
    rule_based_classify,
    Classifier,
    CHAIN_IDENTIFIERS,
    _classification_cache
)
from pharmacy_scraper.classification.data_models import (
    PharmacyData,
    ClassificationResult,
    ClassificationMethod,
    ClassificationSource
)
from pharmacy_scraper.classification.perplexity_client import PerplexityClient

# Helper function to load test data
def load_test_data(filename: str) -> List[Dict]:
    """Load test data from a JSON file in the test_data directory."""
    test_data_dir = Path(__file__).parent / 'test_data'
    test_data_file = test_data_dir / f"{filename}.json"
    
    if not test_data_file.exists():
        return []
        
    with open(test_data_file, 'r') as f:
        return json.load(f)

# Test data
SAMPLE_PHARMACIES = [
    {
        "name": "Downtown Pharmacy",
        "address": "123 Main St",
        "city": "San Francisco",
        "state": "CA",
        "zip": "94105",
        "phone": "(555) 123-4567"
    },
    {
        "name": "CVS Pharmacy #1234",
        "address": "456 Market St",
        "city": "San Francisco",
        "state": "CA",
        "zip": "94103",
        "phone": "(555) 987-6543"
    },
    {
        "name": "Family Care Compounding",
        "address": "789 Oak St",
        "city": "Austin",
        "state": "TX",
        "zip": "73301",
        "phone": "(555) 456-7890"
    },
    {
        "name": "",  # Empty name
        "address": "101 Pine St"
    },
    {
        "name": "ANMC Pharmacy",  # Hospital-affiliated
        "address": "4321 Hospital Dr",
        "city": "Anchorage",
        "state": "AK",
        "zip": "99508"
    },
    {
        "name": "Walgreens #12345",  # Another chain
        "address": "789 Broadway",
        "city": "New York",
        "state": "NY",
        "zip": "10003"
    },
    {
        "name": None,  # None name
        "address": "2020 None St",
        "city": "Nowhere",
        "state": "NW"
    },
    {
        "name": "Specialty RX",  # Minimal data
        "address": None
    }
]

def test_llm_classification():
    """Test LLM-based classification of pharmacies."""
    # Check for required environment variable
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        pytest.skip("PERPLEXITY_API_KEY environment variable not set")
    
    # Check for required packages
    try:
        import openai
        from openai import OpenAI
    except ImportError:
        pytest.skip("openai package not installed")
    
    from pharmacy_scraper.classification import classify_pharmacy
    from pharmacy_scraper.classification.data_models import ClassificationResult
    
    # Test independent pharmacy
    result = classify_pharmacy(SAMPLE_PHARMACIES[0])
    assert isinstance(result, ClassificationResult)
    assert hasattr(result, 'is_chain')
    assert hasattr(result, 'confidence')
    assert 0 <= result.confidence <= 1.0
    
    # Test chain pharmacy
    result = classify_pharmacy(SAMPLE_PHARMACIES[1])  # CVS Pharmacy
    assert result.is_chain is True
    assert result.confidence >= 0.9  # High confidence for chains
    assert "CVS" in result.reason  # Should mention the chain in the reason
    
    # Test hospital-affiliated pharmacy
    result = classify_pharmacy(SAMPLE_PHARMACIES[4])  # ANMC Pharmacy
    assert result.is_chain is True  # Should be classified as chain (hospital system)
    assert result.confidence >= 0.9
    
    # Test another chain
    result = classify_pharmacy(SAMPLE_PHARMACIES[5])  # Walgreens
    assert result.is_chain is True
    assert "Walgreens" in result.reason

def test_rule_based_classification():
    """Test rule-based classification fallback."""
    from pharmacy_scraper.classification.classifier import rule_based_classify, CHAIN_IDENTIFIERS
    
    # Test independent pharmacy
    result = rule_based_classify(SAMPLE_PHARMACIES[0])
    assert result.is_chain is False
    assert result.method == ClassificationMethod.RULE_BASED
    assert result.source == ClassificationSource.RULE_BASED
    assert result.confidence > 0  # Should have some confidence
    
    # Test chain pharmacy (CVS)
    result = rule_based_classify(SAMPLE_PHARMACIES[1])
    assert result.is_chain is True
    assert result.method == ClassificationMethod.RULE_BASED
    assert "CVS" in result.reason
    
    # Test hospital-affiliated pharmacy
    result = rule_based_classify(SAMPLE_PHARMACIES[4])  # ANMC Pharmacy
    assert result.is_chain is True
    assert any(id in result.reason for id in ["ANMC", "hospital"])
    
    # Test with empty name
    result = rule_based_classify(SAMPLE_PHARMACIES[3])  # Empty name
    assert isinstance(result, ClassificationResult)
    assert result.is_chain is False  # Default to independent for empty names
    
    # Test with None name
    result = rule_based_classify(SAMPLE_PHARMACIES[6])  # None name
    assert isinstance(result, ClassificationResult)
    
    # Test with minimal data
    result = rule_based_classify(SAMPLE_PHARMACIES[7])  # Only name
    assert isinstance(result, ClassificationResult)
    
    # Test all chain identifiers
    for chain in CHAIN_IDENTIFIERS:
        test_pharmacy = {"name": f"Test {chain} Pharmacy", "address": "123 Test St"}
        result = rule_based_classify(test_pharmacy)
        assert result.is_chain is True, f"Failed to identify chain: {chain}"

def test_compounding_pharmacy_detection():
    """Test that compounding pharmacies are properly identified as independent."""
    from pharmacy_scraper.classification.classifier import rule_based_classify
    
    # Test with explicit compounding in name
    result = rule_based_classify(SAMPLE_PHARMACIES[2])  # Family Care Compounding
    assert result.is_chain is False
    assert result.is_compounding is True
    assert "compounding" in result.reason.lower()
    assert result.confidence >= 0.9  # High confidence for compounding
    
    # Test with variations of compounding
    test_cases = [
        ("ABC Compounding Pharmacy", True),  # Middle of name
        ("XYZ Compounding", True),  # End of name
        ("Compounding Specialists", True),  # Start of name
        ("Regular Pharmacy", False),  # No compounding
        ("Pharmacy Compounding Solutions", True),  # Middle of name
        ("Non-Compounding Pharmacy", True),  # Contains 'compounding' as whole word
        ("Compounding", True),  # Single word
        ("ABC Compounding", True),  # End of name
        ("Compounding Pharmacy", True),  # Start of name with space
        ("Specialty Compounding Center", True),  # Middle of name
        ("ABC Compounding Solutions", True),  # Multiple words
        ("Non-Compound Pharmacy", False),  # Similar but not matching
        ("Compounders", False),  # Similar but not matching
        ("Recompounding", False)  # Similar but not matching
    ]
    
    for name, expected_compounding in test_cases:
        result = rule_based_classify({"name": name})
        assert result.is_compounding == expected_compounding, f"Failed for: {name}"

def test_batch_classification():
    """Test batch processing of pharmacy classification."""
    # Check for required environment variable
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        pytest.skip("PERPLEXITY_API_KEY environment variable not set")
    
    # Check for required packages
    try:
        import openai
        from openai import OpenAI
    except ImportError:
        pytest.skip("openai package not installed")
        
    from pharmacy_scraper.classification import batch_classify_pharmacies, classify_pharmacy
    from pharmacy_scraper.classification.classifier import _classification_cache
    from pharmacy_scraper.classification.data_models import ClassificationResult, PharmacyData
    
    # Clear cache before test
    _classification_cache.clear()
    
    # Test with all sample pharmacies
    results = batch_classify_pharmacies(SAMPLE_PHARMACIES)
    
    # Basic validation
    assert len(results) == len(SAMPLE_PHARMACIES)
    assert all(isinstance(r, ClassificationResult) for r in results)
    assert all(hasattr(r, 'is_chain') for r in results)
    assert all(hasattr(r, 'confidence') for r in results)
    assert all(0 <= r.confidence <= 1.0 for r in results)
    
    # Verify chain pharmacies are correctly identified
    chain_indices = [1, 4, 5]  # Indices of chain pharmacies in SAMPLE_PHARMACIES
    for idx in chain_indices:
        assert results[idx].is_chain is True, f"Expected chain pharmacy at index {idx}"
    
    # Test with empty list
    empty_results = batch_classify_pharmacies([])
    assert len(empty_results) == 0
    
    # Test with single item
    single_result = batch_classify_pharmacies([SAMPLE_PHARMACIES[0]])
    assert len(single_result) == 1
    assert isinstance(single_result[0], ClassificationResult)
    
    # Test with custom cache
    custom_cache = {}
    test_pharmacies = SAMPLE_PHARMACIES[:2]  # Just first two to be faster
    results_with_cache = batch_classify_pharmacies(
        test_pharmacies,
        cache=custom_cache,
        use_llm=False  # Faster for testing
    )
    
    # Verify cache was populated
    assert len(custom_cache) > 0, "Cache should be populated after batch classification"
    assert len(results_with_cache) == len(test_pharmacies)
    
    # Test that results are cached
    with patch('pharmacy_scraper.classification.classifier.rule_based_classify') as mock_rule_based:
        # This should use the cache, not call rule_based_classify
        results_cached = batch_classify_pharmacies(test_pharmacies, cache=custom_cache, use_llm=False)
        mock_rule_based.assert_not_called()
        assert len(results_cached) == len(test_pharmacies)
    
    # Test with mixed input types (dict and PharmacyData)
    mixed_input = [
        test_pharmacies[0],  # dict
        PharmacyData.from_dict(test_pharmacies[1])  # PharmacyData
    ]
    mixed_results = batch_classify_pharmacies(mixed_input, use_llm=False)
    assert len(mixed_results) == len(mixed_input)
    
    # Test with invalid input
    with pytest.raises(AttributeError):
        batch_classify_pharmacies(["not a pharmacy dict"])

def test_cache_behavior(monkeypatch):
    """Test that caching works as expected."""
    from pharmacy_scraper.classification.classifier import (
        classify_pharmacy,
        _classification_cache,
        _get_cache_key
    )
    
    # Clear cache before test
    _classification_cache.clear()
    
    # Create a test pharmacy
    test_pharmacy = {
        "name": "Test Cache Pharmacy",
        "address": "123 Cache St"
    }
    
    # First call - should not be in cache
    result1 = classify_pharmacy(test_pharmacy, use_llm=False)
    assert _get_cache_key(test_pharmacy, use_llm=False) in _classification_cache
    
    # Mock the rule_based_classify to ensure cache is used
    with patch('pharmacy_scraper.classification.classifier.rule_based_classify') as mock_rule_based:
        # Second call - should use cache, not call rule_based_classify
        result2 = classify_pharmacy(test_pharmacy, use_llm=False)
        mock_rule_based.assert_not_called()
    
    # Results should be the same object (same memory address)
    assert result1 is result2
    
    # Test cache with different use_llm parameter
    _classification_cache.clear()
    result_llm = classify_pharmacy(test_pharmacy, use_llm=True)
    result_no_llm = classify_pharmacy(test_pharmacy, use_llm=False)
    assert result_llm is not result_no_llm  # Different cache keys
    
    # Test with different input types but same data
    pharma_obj = PharmacyData.from_dict(test_pharmacy)
    result_dict = classify_pharmacy(test_pharmacy, use_llm=False)
    result_obj = classify_pharmacy(pharma_obj, use_llm=False)
    assert result_dict.is_chain == result_obj.is_chain
    
    # Test cache invalidation
    cache_key = _get_cache_key(test_pharmacy, use_llm=False)
    _classification_cache[cache_key] = ClassificationResult(
        is_chain=True,  # Different from original
        confidence=1.0,
        reason="Test cache",
        method=ClassificationMethod.RULE_BASED,
        source=ClassificationSource.RULE_BASED
    )
    result_cached = classify_pharmacy(test_pharmacy, use_llm=False)
    assert result_cached.is_chain is True  # Should get cached value


def test_classification_with_mock(monkeypatch):
    """Test classification with mocked LLM responses."""
    # Import the necessary classes
    from pharmacy_scraper.classification.data_models import ClassificationResult, ClassificationMethod, ClassificationSource
    from pharmacy_scraper.classification.classifier import classify_pharmacy, rule_based_classify, _classification_cache
    
    # Clear the cache to ensure we don't get cached results
    _classification_cache.clear()
    
    # Create a mock result with the expected default confidence of 0.75
    mock_result = ClassificationResult(
        is_chain=False,
        confidence=0.75,  # This matches the default in query_perplexity
        reason="Stub LLM result",  # This matches the default in query_perplexity
        method=ClassificationMethod.LLM,
        source=ClassificationSource.PERPLEXITY
    )
    
    # Create a mock rule-based classification that returns low confidence
    def mock_rule_based(pharmacy):
        return ClassificationResult(
            is_chain=False,
            confidence=0.1,  # Very low confidence to force LLM fallback
            reason="Low confidence rule-based result",
            method=ClassificationMethod.RULE_BASED,
            source=ClassificationSource.RULE_BASED
        )
    
    # Patch the rule_based_classify function
    monkeypatch.setattr('pharmacy_scraper.classification.classifier.rule_based_classify', mock_rule_based)
    
    # Patch the query_perplexity function
    with patch('pharmacy_scraper.classification.classifier.query_perplexity') as mock_query:
        # Set up the mock
        mock_query.return_value = mock_result
        
        # Test classification with LLM and a custom cache to avoid interference
        custom_cache = {}
        result = classify_pharmacy(
            SAMPLE_PHARMACIES[0],
            use_llm=True,
            cache=custom_cache
        )
        
        # Verify the mock was called
        mock_query.assert_called_once()
        
        # Verify the result
        assert result.is_chain is False
        assert result.confidence == 0.75  # This should match the default in query_perplexity
        assert result.reason == "Stub LLM result"
        assert result.method == ClassificationMethod.LLM
        assert result.source == ClassificationSource.PERPLEXITY
        
        # Test that cache was populated in the custom cache
        from pharmacy_scraper.classification.classifier import _get_cache_key
        cache_key = _get_cache_key(SAMPLE_PHARMACIES[0], use_llm=True)
        assert cache_key in custom_cache, "Result should be in the custom cache"
        
        # Test that subsequent call with same parameters uses cache
        mock_query.reset_mock()
        result2 = classify_pharmacy(SAMPLE_PHARMACIES[0], use_llm=True, cache=custom_cache)
        mock_query.assert_not_called()
        assert result is result2, "Should be the same object from cache"

def test_confidence_threshold():
    """Test that confidence threshold affects classification results."""
    from pharmacy_scraper.classification.classifier import classify_pharmacy
    
    # Test with a pharmacy that should be classified as a chain with high confidence
    pharmacy = {"name": "CVS Pharmacy", "address": "123 Main St"}
    
    # With default threshold (0.0), should classify as chain
    result = classify_pharmacy(pharmacy, use_llm=False)
    assert result.is_chain is True
    
    # With threshold higher than confidence, should return None
    with pytest.raises(ValueError):
        classify_pharmacy(pharmacy, use_llm=False, min_confidence=1.0)


# Property-based tests using hypothesis
@settings(suppress_health_check=[HealthCheck.too_slow])
@given(
    name=st.text(min_size=1, max_size=100, alphabet=st.characters(
        whitelist_categories=('L', 'N', 'P', 'Zs'),
        # Exclude whitespace-only strings
        blacklist_characters=['\x00', '\n', '\r', '\t', '\x0b', '\x0c']
    )).filter(lambda x: x.strip()),  # Ensure non-whitespace only
    address=st.text(min_size=1, max_size=200, alphabet=st.characters(
        whitelist_categories=('L', 'N', 'P', 'Zs'),
        blacklist_characters=['\x00', '\n', '\r', '\t', '\x0b', '\x0c']
    )).filter(lambda x: x.strip()),  # Ensure non-whitespace only
    is_chain=st.booleans(),
    is_compounding=st.booleans()
)
def test_pharmacy_data_model(name: str, address: str, is_chain: bool, is_compounding: bool):
    """Test that PharmacyData model handles various inputs correctly."""
    # Skip if name or address is empty after stripping
    if not name.strip() or not address.strip():
        pytest.skip("Skipping test with empty name or address")
        
    # Create a pharmacy data instance
    pharmacy = PharmacyData(name=name, address=address)
    
    # Test that the model was created correctly (compare with stripped versions)
    assert pharmacy.name.strip() == name.strip()
    assert pharmacy.address.strip() == address.strip()
    
    # Test classification result creation
    result = ClassificationResult(
        is_chain=is_chain,
        is_compounding=is_compounding,
        confidence=0.8,
        reason="Test reason",
        method=ClassificationMethod.RULE_BASED,
        source=ClassificationSource.RULE_BASED
    )
    
    # Test the classification result
    assert result.is_chain == is_chain
    assert result.is_compounding == is_compounding
    assert 0 <= result.confidence <= 1.0
    assert result.method in ClassificationMethod
    assert result.source in ClassificationSource


def _token_match(text: str, keyword: str) -> bool:
    """Helper function to match whole words in text."""
    pattern = rf'\b{re.escape(keyword.lower())}\b'
    return bool(re.search(pattern, text.lower()))


@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50)
@given(
    name=st.text(
        min_size=1,
        max_size=100,
        alphabet=st.characters(
            whitelist_categories=('L', 'N', 'P', 'Zs'),
            blacklist_characters=['\x00', '\n', '\r', '\t', '\x0b', '\x0c']
        )
    ).filter(lambda x: x.strip() and len(x.strip()) > 1),  # Ensure non-whitespace and minimum length
    use_llm=st.booleans()
)
def test_rule_based_classification_property(name: str, use_llm: bool):
    """Property-based test for rule-based classification."""
    from pharmacy_scraper.classification.classifier import rule_based_classify
    
    # Skip if name is empty after stripping
    if not name.strip():
        pytest.skip("Skipping test with empty name")
    
    # Create test data
    pharmacy = {"name": name, "address": "123 Test St"}
    
    # Get classification result
    result = rule_based_classify(pharmacy)
    
    # Verify basic properties
    assert isinstance(result, ClassificationResult)
    assert hasattr(result, 'is_chain')
    assert hasattr(result, 'is_compounding')
    assert hasattr(result, 'confidence')
    assert hasattr(result, 'reason')
    assert hasattr(result, 'method')
    assert hasattr(result, 'source')
    
    # Verify confidence is within valid range
    assert 0 <= result.confidence <= 1.0
    
    # If the name contains a chain identifier, it should be classified as a chain
    chain_identifiers = ["CVS", "Walgreens", "Rite Aid", "Walmart", "Costco"]
    is_chain = any(_token_match(name, kw) for kw in chain_identifiers)
    
    # If the name contains 'compounding' as a whole word, it should be classified as compounding
    is_compounding = _token_match(name, "compounding")
    
    # Verify the classification matches our expectations
    if is_chain:
        assert result.is_chain is True, f"Expected chain for name: {name!r}"
    if is_compounding:
        assert result.is_compounding is True, f"Expected compounding for name: {name!r}"

def test_edge_cases():
    """Test classification with various edge cases."""
    from pharmacy_scraper.classification.classifier import classify_pharmacy, rule_based_classify, _get_cache_key
    
    # Test with None input
    with pytest.raises(ValueError):
        classify_pharmacy(None)
    
    # Test with empty dict - should raise ValueError
    with pytest.raises(ValueError, match="Pharmacy data cannot be empty"):
        classify_pharmacy({})
    
    # Test with minimal data (only name)
    result = classify_pharmacy({"name": "Minimal Pharmacy"})
    assert isinstance(result, ClassificationResult)
    assert not result.is_chain
    assert not result.is_compounding
    assert 0 < result.confidence <= 1.0
    
    # Test with only address (no name)
    result = classify_pharmacy({"address": "123 Test St"})
    assert isinstance(result, ClassificationResult)
    
    # Test with various whitespace and case variations
    test_cases = [
        ({"name": "  CVS  "}, True),  # Extra whitespace
        ({"name": "cVs pHaRmAcY"}, True),  # Mixed case
        ({"name": "\tWALGREENS\n"}, True),  # Whitespace characters
    ]
    
    for data, expected_chain in test_cases:
        result = rule_based_classify(data)
        assert result.is_chain == expected_chain, f"Failed for: {data}"
    
    # Test cache key generation with different input types
    pharma_data = PharmacyData(name="Test", address="123 St")
    dict_data = {"name": "Test", "address": "123 St"}
    
    key1 = _get_cache_key(pharma_data, use_llm=True)
    key2 = _get_cache_key(dict_data, use_llm=True)
    assert key1 == key2, "Same pharmacy data should generate same cache key regardless of input type"
    
    # Test cache key differs with use_llm
    key3 = _get_cache_key(pharma_data, use_llm=False)
    assert key1 != key3, "Cache key should differ based on use_llm parameter"
    
    # Test with non-dict, non-PharmacyData input
    with pytest.raises(AttributeError):
        classify_pharmacy("not a pharmacy")
    
    # Test with PharmacyData object
    pharma_data = PharmacyData(
        name="Test Pharmacy",
        address="123 Test St",
        phone="(555) 123-4567"
    )
    result = classify_pharmacy(pharma_data)
    assert isinstance(result, ClassificationResult)
    
    # Test with PharmacyData from dict
    pharma_from_dict = PharmacyData.from_dict({"name": "From Dict"})
    result = classify_pharmacy(pharma_from_dict)
    assert isinstance(result, ClassificationResult)


def test_confidence_threshold(caplog, monkeypatch):
    """Test that classification results respect confidence thresholds."""
    # Set up logging
    caplog.set_level(logging.DEBUG)
    
    # Disable cache for this test to avoid interference
    monkeypatch.setattr('pharmacy_scraper.classification.classifier._classification_cache', {})
    
    # Create a mock classifier instance
    mock_classifier_instance = MagicMock()
    
    with patch("pharmacy_scraper.classification.classifier.query_perplexity") as mock_llm, \
         patch("pharmacy_scraper.classification.classifier.Classifier") as mock_classifier:
        
        # Set up the mock classifier to return our mock instance
        mock_classifier.return_value = mock_classifier_instance
        
        # Get the rule-based result first to know its confidence
        from pharmacy_scraper.classification.classifier import rule_based_classify
        rule_based_result = rule_based_classify(SAMPLE_PHARMACIES[0])
        
        # Test case 1: LLM confidence is higher than rule-based
        higher_llm_confidence = rule_based_result.confidence + 0.1
        mock_result = ClassificationResult(
            is_chain=False,
            confidence=higher_llm_confidence,
            reason="High confidence LLM result",
            method=ClassificationMethod.LLM,
            source=ClassificationSource.PERPLEXITY
        )
        
        # Reset mocks
        mock_llm.reset_mock()
        mock_classifier_instance.classify_pharmacy.reset_mock()
        
        # Set up mocks
        mock_llm.return_value = mock_result
        mock_classifier_instance.classify_pharmacy.return_value = mock_result
        
        # Debug: Print what we're setting up
        print("\n=== Test Case 1: LLM confidence higher ===")
        print(f"Rule-based confidence: {rule_based_result.confidence}")
        print(f"Mock LLM confidence: {mock_result.confidence}")
        
        # Call the function under test
        result = classify_pharmacy(
            SAMPLE_PHARMACIES[0],
            use_llm=True,
            llm_client=mock_classifier_instance
        )
        
        # Debug: Print what we got
        print(f"Result confidence: {result.confidence}")
        print(f"Result source: {result.source}")
        print(f"Result method: {result.method}")
        print(f"Result reason: {result.reason}")
        
        # Should return LLM result since its confidence is higher
        assert result.is_chain == mock_result.is_chain, \
            f"Expected is_chain={mock_result.is_chain} but got {result.is_chain}"
        assert result.confidence == mock_result.confidence, \
            f"Expected confidence {mock_result.confidence:.2f} but got {result.confidence:.2f}"
        
        # Test case 2: LLM confidence is equal to rule-based
        # The implementation uses >= comparison, so it will return the LLM result when confidences are equal
        equal_llm_confidence = rule_based_result.confidence
        mock_result = ClassificationResult(
            is_chain=False,  # Different from rule-based to make it obvious which result was used
            confidence=equal_llm_confidence,
            reason="Equal confidence LLM result",
            method=ClassificationMethod.LLM,
            source=ClassificationSource.PERPLEXITY
        )
        
        # Reset mocks
        mock_llm.reset_mock()
        mock_classifier_instance.classify_pharmacy.reset_mock()
        
        # Set up mocks
        mock_llm.return_value = mock_result
        mock_classifier_instance.classify_pharmacy.return_value = mock_result
        
        # Clear the cache before this test case
        monkeypatch.setattr('pharmacy_scraper.classification.classifier._classification_cache', {})
        
        result = classify_pharmacy(
            SAMPLE_PHARMACIES[0],
            use_llm=True,
            llm_client=mock_classifier_instance
        )
        
        # Implementation uses >= comparison, so it should return the LLM result when confidences are equal
        assert result.is_chain == mock_result.is_chain, \
            f"Expected is_chain={mock_result.is_chain} but got {result.is_chain}"
        assert result.confidence == mock_result.confidence, \
            f"Expected confidence {mock_result.confidence:.2f} but got {result.confidence:.2f}"
        assert result.method == ClassificationMethod.LLM, \
            f"Expected method=LLM but got {result.method}"
        
        # Test case 3: LLM confidence is lower than rule-based
        lower_llm_confidence = max(0.0, rule_based_result.confidence - 0.1)
        mock_result = ClassificationResult(
            is_chain=not rule_based_result.is_chain,  # Different from rule-based
            confidence=lower_llm_confidence,
            reason="Low confidence LLM result",
            method=ClassificationMethod.LLM,
            source=ClassificationSource.PERPLEXITY
        )
        
        # Reset mocks
        mock_llm.reset_mock()
        mock_classifier_instance.classify_pharmacy.reset_mock()
        
        # Set up mocks
        mock_llm.return_value = mock_result
        mock_classifier_instance.classify_pharmacy.return_value = mock_result
        
        # Clear the cache before this test case
        monkeypatch.setattr('pharmacy_scraper.classification.classifier._classification_cache', {})
        
        result = classify_pharmacy(
            SAMPLE_PHARMACIES[0],
            use_llm=True,
            llm_client=mock_classifier_instance
        )
        
        # Should return rule-based result since LLM confidence is lower
        assert result.is_chain == rule_based_result.is_chain, \
            f"Expected is_chain={rule_based_result.is_chain} but got {result.is_chain}"
        assert result.confidence == rule_based_result.confidence, \
            f"Expected confidence {rule_based_result.confidence:.2f} but got {result.confidence:.2f}"
        assert result.method == ClassificationMethod.RULE_BASED, \
            f"Expected method=RULE_BASED but got {result.method}"
