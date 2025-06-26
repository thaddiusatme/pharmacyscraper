"""
Tests for the pharmacy classification module.
"""
import pytest
import logging
from unittest.mock import patch, MagicMock, call
import pandas as pd
from typing import Dict, Any

from pharmacy_scraper.classification.classifier import (
    classify_pharmacy,
    batch_classify_pharmacies,
    rule_based_classify
)
from pharmacy_scraper.classification.perplexity_client import PerplexityClient

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
    }
]

def test_llm_classification():
    """Test LLM-based classification of pharmacies."""
    # Skip if Perplexity API key is not available
    pytest.importorskip("perplexity")
    
    from pharmacy_scraper.classification import classify_pharmacy
    
    # Test independent pharmacy
    result = classify_pharmacy(SAMPLE_PHARMACIES[0])
    assert "is_chain" in result
    assert "confidence" in result
    assert 0 <= result["confidence"] <= 1.0
    
    # Test chain pharmacy
    result = classify_pharmacy(SAMPLE_PHARMACIES[1])
    assert result["is_chain"] is True
    assert result["confidence"] >= 0.9  # High confidence for chains

def test_rule_based_classification():
    """Test rule-based classification fallback."""
    from pharmacy_scraper.classification.classifier import rule_based_classify
    
    # Test independent pharmacy
    result = rule_based_classify(SAMPLE_PHARMACIES[0])
    assert result.is_chain is False
    assert result.method == "rule_based"
    
    # Test chain pharmacy
    result = rule_based_classify(SAMPLE_PHARMACIES[1])
    assert result.is_chain is True
    assert result.method == "rule_based"

def test_compounding_pharmacy_detection():
    """Test that compounding pharmacies are properly identified as independent."""
    from pharmacy_scraper.classification.classifier import rule_based_classify
    
    result = rule_based_classify(SAMPLE_PHARMACIES[2])
    assert result.is_chain is False
    assert "compounding" in result.reason.lower()

def test_batch_classification():
    """Test batch processing of pharmacy classification."""
    pytest.importorskip("perplexity")
    from pharmacy_scraper.classification import batch_classify_pharmacies
    
    df = pd.DataFrame(SAMPLE_PHARMACIES)
    results = batch_classify_pharmacies(df)
    
    assert len(results) == len(SAMPLE_PHARMACIES)
    assert all(col in results.columns for col in ["is_chain", "confidence", "classification_method"])
    assert results["is_chain"].dtype == bool

def test_classification_with_mock():
    """Test classification with mocked LLM responses."""
    with patch("pharmacy_scraper.classification.classifier.query_perplexity") as mock_llm:
        # Import ClassificationResult to use in the mock
        from pharmacy_scraper.classification.data_models import ClassificationResult, ClassificationMethod, ClassificationSource
        
        # Create a mock ClassificationResult object
        mock_result = ClassificationResult(
            is_chain=False,
            confidence=0.95,
            reason="Independent based on name and lack of chain indicators",
            method=ClassificationMethod.LLM,
            source=ClassificationSource.PERPLEXITY
        )
        mock_llm.return_value = mock_result
        
        from pharmacy_scraper.classification.classifier import classify_pharmacy
        result = classify_pharmacy(SAMPLE_PHARMACIES[0])
        
        assert result.is_chain is False
        assert result.confidence >= 0.9
        assert hasattr(result, 'reason')
        mock_llm.assert_called_once()

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
        # Import ClassificationResult to use in the mock
        from pharmacy_scraper.classification.data_models import ClassificationResult, ClassificationMethod, ClassificationSource
        
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
