"""
Enhanced tests for the pharmacy classification module.

This module contains additional test cases to improve test coverage
and ensure robustness of the classification system.
"""
import pytest
import re
from unittest.mock import patch, MagicMock
from typing import Dict, Any, Optional

from pharmacy_scraper.classification.classifier import (
    classify_pharmacy,
    rule_based_classify,
    _get_cache_key,
    Classifier,
    CHAIN_IDENTIFIERS
)
from pharmacy_scraper.classification.data_models import (
    PharmacyData,
    ClassificationResult,
    ClassificationMethod,
    ClassificationSource,
    DEFAULT_INDEPENDENT
)

# Test data for hospital-affiliated pharmacies
HOSPITAL_PHARMACIES = [
    {"name": "ANMC Pharmacy", "address": "123 Hospital Dr"},
    {"name": "VA Medical Center Pharmacy", "address": "456 Veteran Ave"},
    {"name": "Kaiser Permanente Pharmacy", "address": "789 Health St"},
    {"name": "Clinic Pharmacy", "address": "101 Care Ln"},
    {"name": "Medical Center Pharmacy", "address": "202 Wellness Blvd"},
]

# Test data for edge cases
EDGE_CASE_PHARMACIES = [
    ({"name": "", "address": "123 Main St"}, "Empty name"),
    ({"name": " ", "address": " "}, "Whitespace only"),
    ({"name": "A", "address": "B"}, "Minimal data"),
    ({"name": "!@#$%^&*()", "address": "!@#$%^&*()"}, "Special chars"),
    ({"name": "CVS", "address": ""}, "Chain with empty address"),
]

# Test data for compounding pharmacies
COMPOUNDING_PHARMACIES = [
    {"name": "Compounding Pharmacy", "address": "123 Main St"},
    {"name": "Custom Compounding Center", "address": "456 Oak Ave"},
    {"name": "Specialty Compounding Pharmacy", "address": "789 Pine St"},
]

def test_hospital_affiliated_pharmacies():
    """Test that hospital-affiliated pharmacies are properly identified."""
    for pharmacy in HOSPITAL_PHARMACIES:
        result = rule_based_classify(pharmacy)
        assert result.is_chain is True, f"Expected {pharmacy['name']} to be classified as a chain"
        # Check if the reason contains any of the expected patterns or the chain keyword
        assert ("chain" in result.reason.lower() or 
                "hospital" in result.reason.lower() or
                "health system" in result.reason.lower() or 
                "clinic" in result.reason.lower()), \
               f"Expected chain/hospital/clinic in reason for {pharmacy['name']}, got: {result.reason}"

def test_compounding_pharmacies():
    """Test that compounding pharmacies are properly identified as independent."""
    for pharmacy in COMPOUNDING_PHARMACIES:
        result = rule_based_classify(pharmacy)
        assert result.is_compounding is True, f"Expected {pharmacy['name']} to be compounding"
        assert result.is_chain is False, f"Expected {pharmacy['name']} to be independent"
        assert "compounding" in result.reason.lower(), \
               f"Expected 'compounding' in reason for {pharmacy['name']}"

def test_edge_case_handling():
    """Test classification with various edge cases."""
    for pharmacy, description in EDGE_CASE_PHARMACIES:
        try:
            result = rule_based_classify(pharmacy)
            assert isinstance(result, ClassificationResult), \
                   f"Expected ClassificationResult for {description}"
            assert hasattr(result, 'is_chain'), \
                   f"Result should have is_chain attribute for {description}"
            assert hasattr(result, 'confidence'), \
                   f"Result should have confidence attribute for {description}"
        except Exception as e:
            pytest.fail(f"Unexpected error for {description}: {str(e)}")

def test_cache_key_generation():
    """Test cache key generation with various input types."""
    # Test with dict input
    pharma_dict = {"name": "Test", "address": "123 St"}
    key1 = _get_cache_key(pharma_dict)
    
    # Test with PharmacyData input
    pharma_obj = PharmacyData(name="Test", address="123 St")
    key2 = _get_cache_key(pharma_obj)
    
    # Same data should generate same key regardless of input type
    assert key1 == key2, "Same pharmacy data should generate same cache key"
    
    # Different use_llm should generate different keys
    key3 = _get_cache_key(pharma_dict, use_llm=False)
    assert key1 != key3, "Different use_llm should generate different keys"
    
    # Different data should generate different keys
    pharma_diff = {"name": "Different", "address": "456 St"}
    key4 = _get_cache_key(pharma_diff)
    assert key1 != key4, "Different pharmacy data should generate different keys"

def test_classifier_initialization():
    """Test Classifier initialization with and without client."""
    # Test with no client (should try to create one)
    with patch('pharmacy_scraper.classification.perplexity_client.PerplexityClient') as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        
        classifier = Classifier()
        assert classifier._client is not None
        mock_client_cls.assert_called_once()
    
    # Test with explicit None client (should still try to create one)
    with patch('pharmacy_scraper.classification.perplexity_client.PerplexityClient') as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        
        classifier = Classifier(client=None)
        assert classifier._client is not None
        mock_client_cls.assert_called_once()
    
    # Test with mock client
    mock_client = MagicMock()
    classifier = Classifier(client=mock_client)
    assert classifier._client is mock_client

def test_classifier_classify_pharmacy():
    """Test Classifier.classify_pharmacy method."""
    # Create a mock LLM client
    mock_client = MagicMock()
    
    # Set up the mock LLM client to return a known result
    mock_client.classify_pharmacy.return_value = ClassificationResult(
        is_chain=False,
        is_compounding=False,
        confidence=0.9,
        reason="Mock LLM result",
        method=ClassificationMethod.LLM,
        source=ClassificationSource.PERPLEXITY
    )
    
    # Create a mock rule-based result
    mock_rule_result = ClassificationResult(
        is_chain=False,
        is_compounding=False,
        confidence=0.5,
        reason="Rule-based result",
        method=ClassificationMethod.RULE_BASED,
        source=ClassificationSource.RULE_BASED
    )
    
    # Patch the rule_based_classify function
    with patch('pharmacy_scraper.classification.classifier.rule_based_classify', 
              return_value=mock_rule_result) as mock_rule_based:
        
        classifier = Classifier(client=mock_client)
        
        # Test with LLM enabled (should use LLM since rule-based confidence is 0.5 < 0.9)
        result = classifier.classify_pharmacy({"name": "Test", "address": "123 St"}, use_llm=True)
        
        # Verify LLM was called and result is as expected
        mock_client.classify_pharmacy.assert_called_once()
        assert result.method == ClassificationMethod.LLM
        assert result.reason == "Mock LLM result"
        
        # Reset mocks for the next test
        mock_rule_based.reset_mock()
        mock_client.classify_pharmacy.reset_mock()
        
        # Test with LLM disabled (should use rule-based)
        result = classifier.classify_pharmacy({"name": "Test2", "address": "456 St"}, use_llm=False)
        
        # Verify rule-based was called and LLM was not called
        mock_rule_based.assert_called_once()
        mock_client.classify_pharmacy.assert_not_called()
        assert result.method == ClassificationMethod.RULE_BASED
        assert result.reason == "Rule-based result"

def test_chain_pharmacy_detection():
    """Test detection of chain pharmacies."""
    for chain in CHAIN_IDENTIFIERS:
        # Skip hospital identifiers as they're tested separately
        if chain.lower() in ['hospital', 'clinic', 'va', 'medical center', 'health system']:
            continue
            
        # Test with just the chain name
        result = rule_based_classify({"name": f"{chain} Pharmacy", "address": "123 St"})
        assert result.is_chain is True, f"Expected {chain} to be detected as chain"
        
        # Test with additional text
        result = rule_based_classify({"name": f"{chain} #1234", "address": "123 St"})
        assert result.is_chain is True, f"Expected {chain} #1234 to be detected as chain"

def test_independent_pharmacy_detection():
    """Test that independent pharmacies are properly identified."""
    independent_pharmacies = [
        {"name": "Corner Drug Store", "address": "123 Main St"},
        {"name": "Family Pharmacy", "address": "456 Oak Ave"},
        {"name": "Neighborhood Drugs", "address": "789 Pine St"},
    ]
    
    for pharmacy in independent_pharmacies:
        result = rule_based_classify(pharmacy)
        assert result.is_chain is False, f"Expected {pharmacy['name']} to be independent"
        assert result.confidence >= 0.5, f"Expected confidence >= 0.5 for {pharmacy['name']}"
        # Accept either 'no chain identifiers' or 'independent' in the reason
        assert ("no chain identifiers" in result.reason.lower() or 
                "independent" in result.reason.lower()), \
               f"Expected 'no chain identifiers' or 'independent' in reason for {pharmacy['name']}, got: {result.reason}"

def test_llm_fallback_behavior(monkeypatch):
    """Test that LLM fallback works when rule-based has low confidence."""
    # Mock the LLM to return a known result
    mock_result = ClassificationResult(
        is_chain=False,
        is_compounding=False,
        confidence=0.8,
        reason="LLM fallback",
        method=ClassificationMethod.LLM,
        source=ClassificationSource.PERPLEXITY
    )
    
    # Patch the query_perplexity function
    with patch('pharmacy_scraper.classification.classifier.query_perplexity', 
              return_value=mock_result) as mock_query:
        # Test with a pharmacy that would have low confidence from rule-based
        result = classify_pharmacy({"name": "Ambiguous Pharmacy", "address": "123 St"}, use_llm=True)
        
        # Should have used the LLM
        mock_query.assert_called_once()
        
        # Should return the LLM result
        assert result == mock_result
        assert result.reason == "LLM fallback"
        assert result.method == ClassificationMethod.LLM

def test_batch_classification_with_edge_cases():
    """Test batch classification with various edge cases."""
    from pharmacy_scraper.classification.classifier import batch_classify_pharmacies
    
    # Create test cases with various scenarios
    test_cases = [
        {"name": "CVS Pharmacy", "address": "123 Main St"},  # Chain
        {"name": "Corner Drug", "address": "456 Oak Ave"},   # Independent
        {"name": "", "address": "789 Pine St"},              # Empty name
        {"name": "ANMC Pharmacy", "address": "101 Hospital Dr"},  # Hospital
        {"name": "Compounding Specialists", "address": "202 Main St"},  # Compounding
    ]
    
    results = batch_classify_pharmacies(test_cases, use_llm=False)
    
    # Should return one result per input
    assert len(results) == len(test_cases)
    
    # All results should be ClassificationResult objects
    assert all(isinstance(r, ClassificationResult) for r in results)
    
    # Verify expected classifications
    assert results[0].is_chain is True  # CVS
    assert results[1].is_chain is False  # Independent
    assert results[3].is_chain is True  # Hospital
    assert results[4].is_compounding is True  # Compounding
