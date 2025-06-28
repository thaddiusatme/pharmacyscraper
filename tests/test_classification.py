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

@pytest.fixture
def mock_perplexity_client():
    """Provides a mocked PerplexityClient instance."""
    with patch("pharmacy_scraper.classification.perplexity_client.PerplexityClient") as MockPerplexityClient:
        mock_instance = MockPerplexityClient.return_value
        yield mock_instance

def test_llm_classification(mock_perplexity_client, monkeypatch):
    # Ensure a clean cache for this test
    monkeypatch.setattr('pharmacy_scraper.classification.classifier._classification_cache', {})
    """Test LLM-based classification using a mocked client."""
    # Arrange: Configure the mock to return a specific result
    mock_result = ClassificationResult(
        is_chain=False,
        confidence=0.85,
        reason="LLM determined this is an independent pharmacy.",
        method=ClassificationMethod.LLM,
        source=ClassificationSource.PERPLEXITY
    )
    mock_perplexity_client.classify_pharmacy.return_value = mock_result

    # Act: Instantiate the classifier and call the method
    classifier = Classifier(client=mock_perplexity_client)
    pharmacy_data = SAMPLE_PHARMACIES[0]  # "Downtown Pharmacy"
    result = classifier.classify_pharmacy(pharmacy_data)

    # Assert: Verify the mock was called correctly and the result is as expected
    mock_perplexity_client.classify_pharmacy.assert_called_once_with(pharmacy_data)
    assert result == mock_result
    assert result.is_chain is False
    assert result.confidence == 0.85

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

def test_batch_classification(mock_perplexity_client, monkeypatch):
    """Test batch classification of pharmacies."""
    # Ensure a clean cache for this test
    monkeypatch.setattr('pharmacy_scraper.classification.classifier._classification_cache', {})

    mock_perplexity_client.classify_pharmacy.return_value = ClassificationResult(
        is_chain=False, confidence=0.9, reason="LLM", method=ClassificationMethod.LLM
    )
    
    classifier = Classifier(client=mock_perplexity_client)
    results = [classifier.classify_pharmacy(p) for p in SAMPLE_PHARMACIES]
    
    assert len(results) == len(SAMPLE_PHARMACIES)
    assert all(isinstance(r, ClassificationResult) for r in results)
    # The rule-based classifier has high confidence for most pharmacies in the
    # sample data. Only 'Downtown Pharmacy' should trigger the LLM.
    assert mock_perplexity_client.classify_pharmacy.call_count == 1


def test_classification_with_mock(mock_perplexity_client, monkeypatch):
    # Ensure a clean cache for this test
    monkeypatch.setattr('pharmacy_scraper.classification.classifier._classification_cache', {})
    """Test classification with a mocked LLM client."""
    mock_result = ClassificationResult(
        is_chain=False,
        is_compounding=True,
        confidence=0.95,
        reason="LLM says compounding",
        method=ClassificationMethod.LLM,
    )
    mock_perplexity_client.classify_pharmacy.return_value = mock_result
    
    classifier = Classifier(client=mock_perplexity_client)
    # Use a pharmacy that would be classified as independent by rules
    result = classifier.classify_pharmacy({"name": "A Normal Pharmacy"})
    
    # The LLM result has higher confidence, so it should be returned
    assert result == mock_result
    mock_perplexity_client.classify_pharmacy.assert_called_once()

def test_cache_behavior(mock_perplexity_client, monkeypatch):
    """Test that the cache is used correctly."""
    # Clear cache for this test
    cache = {}
    monkeypatch.setattr('pharmacy_scraper.classification.classifier._classification_cache', cache)
    
    mock_result = ClassificationResult(is_chain=True, confidence=0.99, method=ClassificationMethod.LLM)
    mock_perplexity_client.classify_pharmacy.return_value = mock_result
    
    classifier = Classifier(client=mock_perplexity_client)
    pharmacy_data = {"name": "Test Pharmacy"}
    
    # First call, should call the client
    result1 = classifier.classify_pharmacy(pharmacy_data)
    assert result1.method == ClassificationMethod.LLM
    mock_perplexity_client.classify_pharmacy.assert_called_once()
    
    # Second call, should hit the cache
    result2 = classifier.classify_pharmacy(pharmacy_data)
    assert result2.method == ClassificationMethod.CACHED # The classifier should wrap it
    # The mock should still have been called only once
    mock_perplexity_client.classify_pharmacy.assert_called_once()
    assert result2.is_chain == result1.is_chain

def test_edge_cases():
    """Test edge cases for the classifier."""
    classifier = Classifier(client=None) # No LLM client
    
    # Test with None
    with pytest.raises(ValueError, match="Pharmacy data cannot be None"):
        classifier.classify_pharmacy(None)
        
    result = classifier.classify_pharmacy({})
    assert isinstance(result, ClassificationResult)

    # Test with PharmacyData object
    pharma_data = PharmacyData(name="Test Pharmacy", address="123 Test St")
    result = classifier.classify_pharmacy(pharma_data)
    assert isinstance(result, ClassificationResult)

def test_confidence_threshold(monkeypatch):
    """Test that classification results respect confidence thresholds."""
    # Clear cache
    monkeypatch.setattr('pharmacy_scraper.classification.classifier._classification_cache', {})
    
    pharmacy_data = {"name": "A Pharmacy"}
    rule_based_result = rule_based_classify(pharmacy_data) # confidence 0.5
    
    # Case 1: LLM confidence is higher
    mock_client_high_conf = MagicMock()
    llm_result_high = ClassificationResult(is_chain=True, confidence=0.9, method=ClassificationMethod.LLM)
    mock_client_high_conf.classify_pharmacy.return_value = llm_result_high
    
    classifier_high = Classifier(client=mock_client_high_conf)
    result_high = classifier_high.classify_pharmacy(pharmacy_data)
    
    assert result_high.method == ClassificationMethod.LLM
    assert result_high.confidence == 0.9

    # Case 2: LLM confidence is lower
    monkeypatch.setattr('pharmacy_scraper.classification.classifier._classification_cache', {}) # clear cache again
    mock_client_low_conf = MagicMock()
    llm_result_low = ClassificationResult(is_chain=True, confidence=0.2, method=ClassificationMethod.LLM)
    mock_client_low_conf.classify_pharmacy.return_value = llm_result_low

    classifier_low = Classifier(client=mock_client_low_conf)
    result_low = classifier_low.classify_pharmacy(pharmacy_data)

    assert result_low.method == ClassificationMethod.RULE_BASED
    assert result_low.confidence == rule_based_result.confidence
