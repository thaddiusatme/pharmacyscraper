"""
Tests for the pharmacy classification module.

This module contains unit tests for the pharmacy classification functionality,
including both rule-based and LLM-based classification.
"""

from unittest.mock import patch, MagicMock
import pytest
import json
from pathlib import Path

from pharmacy_scraper.classification.classifier import (
    Classifier,
    rule_based_classify,
    classify_pharmacy,
    batch_classify_pharmacies,
    CHAIN_IDENTIFIERS
)
from pharmacy_scraper.classification.perplexity_client import PerplexityClient, PerplexityAPIError

# Sample pharmacy data for testing
SAMPLE_PHARMACY = {
    "name": "Test Pharmacy",
    "address": "123 Main St, Test City, CA 12345",
    "phone": "(555) 123-4567",
    "website": "https://testpharmacy.com"
}

SAMPLE_CHAIN_PHARMACY = {
    "name": "CVS Pharmacy #1234",
    "address": "456 Oak St, Test City, CA 12345",
    "phone": "(555) 987-6543",
    "website": "https://www.cvs.com"
}

SAMPLE_COMPOUNDING_PHARMACY = {
    "name": "Test Compounding Pharmacy",
    "address": "789 Pine St, Test City, CA 12345",
    "phone": "(555) 456-7890",
    "website": "https://testcompounding.com"
}


class TestRuleBasedClassifier:
    """Tests for the rule-based classification functionality."""

    def test_identifies_chain_pharmacies(self):
        """Test that known chain pharmacies are correctly identified."""
        for chain in CHAIN_IDENTIFIERS[:5]:  # Test first 5 chains
            pharmacy = {"name": f"{chain} #1234", "address": "123 Test St"}
            result = rule_based_classify(pharmacy)
            assert result["is_chain"] is True
            assert result["confidence"] > 0.9
            assert chain in result["reason"]

    def test_identifies_compounding_pharmacies(self):
        """Test that compounding pharmacies are correctly identified."""
        result = rule_based_classify(SAMPLE_COMPOUNDING_PHARMACY)
        assert result["is_chain"] is False
        assert "compounding" in result["reason"].lower()
        assert result["confidence"] >= 0.9

    def test_defaults_to_independent(self):
        """Test that pharmacies without chain indicators are marked as independent."""
        result = rule_based_classify(SAMPLE_PHARMACY)
        assert result["is_chain"] is False
        assert "no chain identifiers" in result["reason"].lower()
        assert 0.5 <= result["confidence"] < 1.0


class TestClassifierIntegration:
    """Integration tests for the main Classifier class."""

    @patch('src.classification.classifier.PerplexityClient')
    def test_classify_uses_rule_based_first(self, mock_client_class):
        """Test that the classifier tries rule-based classification first."""
        # Setup
        mock_client = MagicMock(spec=PerplexityClient)
        mock_client_class.return_value = mock_client
        
        classifier = Classifier(client=mock_client)
        
        # Test with a compounding pharmacy (should be caught by rules)
        result = classifier.classify_pharmacy(SAMPLE_COMPOUNDING_PHARMACY)
        
        # Verify Perplexity wasn't called
        mock_client.classify_pharmacy.assert_not_called()
        assert result["is_compounding"] is True
        assert result["source"] == "rule-based"

    @patch('src.classification.classifier.PerplexityClient')
    def test_falls_back_to_llm_when_needed(self, mock_client_class):
        """Test that the classifier falls back to LLM when rules don't apply."""
        # Setup mock LLM response
        mock_client = MagicMock(spec=PerplexityClient)
        mock_client.classify_pharmacy.return_value = {
            "is_chain": False,
            "confidence": 0.85,
            "reason": "Appears to be an independent pharmacy"
        }
        mock_client_class.return_value = mock_client
        
        classifier = Classifier(client=mock_client)
        result = classifier.classify_pharmacy(SAMPLE_PHARMACY)
        
        # Verify LLM was called
        mock_client.classify_pharmacy.assert_called_once()
        assert result["is_chain"] is False
        assert result["confidence"] == 0.85


class TestClassifyPharmacyFunction:
    """Tests for the classify_pharmacy function."""

    @patch('src.classification.classifier.query_perplexity')
    @patch('src.classification.classifier.rule_based_classify')
    def test_uses_both_classifiers_by_default(self, mock_rule_based, mock_llm):
        """Test that both classifiers are used by default."""
        # Setup mocks
        mock_rule_based.return_value = {
            "is_chain": False,
            "confidence": 0.7,
            "reason": "No chain indicators",
            "method": "rule_based"
        }
        
        mock_llm.return_value = {
            "is_chain": False,
            "confidence": 0.8,
            "reason": "Appears independent",
            "method": "llm"
        }
        
        result = classify_pharmacy(SAMPLE_PHARMACY)
        
        # Verify both classifiers were called
        mock_rule_based.assert_called_once()
        mock_llm.assert_called_once()
        
        # Should prefer the higher confidence result
        assert result["confidence"] == 0.8
        assert result["method"] == "llm"

    @patch('src.classification.classifier.query_perplexity')
    @patch('src.classification.classifier.rule_based_classify')
    def test_skips_llm_when_disabled(self, mock_rule_based, mock_llm):
        """Test that LLM can be disabled."""
        mock_rule_based.return_value = {
            "is_chain": False,
            "confidence": 0.7,
            "reason": "No chain indicators",
            "method": "rule_based"
        }
        
        result = classify_pharmacy(SAMPLE_PHARMACY, use_llm=False)
        
        # Only rule-based should be called
        mock_rule_based.assert_called_once()
        mock_llm.assert_not_called()
        
        assert result["method"] == "rule_based"


class TestBatchClassification:
    """Tests for batch classification functionality."""

    @patch('src.classification.classifier.classify_pharmacy')
    def test_batch_classification(self, mock_classify):
        """Test that batch classification processes multiple pharmacies."""
        # Setup mock to return different results for different inputs
        def side_effect(pharmacy, **kwargs):
            if pharmacy["name"] == "Pharmacy A":
                return {"is_chain": False, "confidence": 0.9}
            return {"is_chain": True, "confidence": 0.95}
            
        mock_classify.side_effect = side_effect
        
        pharmacies = [
            {"name": "Pharmacy A", "address": "123 Test St"},
            {"name": "CVS #123", "address": "456 Test St"}
        ]
        
        results = batch_classify_pharmacies(pharmacies)
        
        assert len(results) == 2
        assert results[0]["is_chain"] is False
        assert results[1]["is_chain"] is True
        assert mock_classify.call_count == 2


class TestErrorHandling:
    """Tests for error handling in the classification pipeline."""

    @patch('src.classification.classifier.PerplexityClient')
    def test_llm_error_handling(self, mock_client_class):
        """Test that LLM errors are handled gracefully."""
        # Setup mock client
        mock_client = MagicMock(spec=PerplexityClient)
        mock_client.classify_pharmacy.side_effect = PerplexityAPIError("API Error")
        mock_client_class.return_value = mock_client
        
        # Create classifier with mock client
        classifier = Classifier(client=mock_client)
        
        # Patch the cache decorator to not interfere with our test
        original_classify = classifier.classify_pharmacy
        classifier.classify_pharmacy = original_classify.__wrapped__.__get__(classifier, Classifier)
        
        # This should not raise an exception
        result = classifier.classify_pharmacy(SAMPLE_PHARMACY)
        
        # Verify LLM was called
        mock_client.classify_pharmacy.assert_called_once_with(SAMPLE_PHARMACY)
        
        # Should return default error result from LLM with source and method
        assert result == {
            "is_chain": False,
            "confidence": 0.0,
            "source": "llm_error",
            "method": "error"
        }

    def test_invalid_input_handling(self):
        """Test that invalid input is handled gracefully."""
        # Create a classifier with a mock client
        mock_client = MagicMock()
        classifier = Classifier(client=mock_client)
        
        # Patch the cache decorator to not interfere with our test
        original_classify = classifier.classify_pharmacy
        classifier.classify_pharmacy = original_classify.__wrapped__.__get__(classifier, Classifier)
        
        # Test with empty input
        result = classifier.classify_pharmacy({})
        
        # Should return default values for invalid input
        assert result == {
            "is_compounding": False,
            "confidence": 0.0,
            "source": "invalid_input",
            "method": "rule_based"
        }
        
        # Should not call the LLM
        mock_client.classify_pharmacy.assert_not_called()


class TestCaching:
    """Tests for the caching functionality."""

    @patch('src.classification.classifier.PerplexityClient')
    def test_cache_hit_returns_cached_result(self, mock_client_class):
        """Test that cached results are returned when available."""
        # Setup mock client
        mock_client = MagicMock(spec=PerplexityClient)
        mock_client.classify_pharmacy.return_value = {
            "is_chain": False,
            "confidence": 0.8,
            "reason": "Test reason",
            "method": "llm"
        }
        mock_client_class.return_value = mock_client
        
        # Create classifier with mock client
        classifier = Classifier(client=mock_client)
        
        # First call - should call classify_pharmacy
        result1 = classifier.classify_pharmacy(SAMPLE_PHARMACY)
        
        # Verify classify_pharmacy was called
        mock_client.classify_pharmacy.assert_called_once_with(SAMPLE_PHARMACY)
        
        # Reset the mock to track new calls
        mock_client.classify_pharmacy.reset_mock()
        
        # Second call with same parameters - should use cache
        result2 = classifier.classify_pharmacy(SAMPLE_PHARMACY)
        
        # Should NOT call classify_pharmacy again
        mock_client.classify_pharmacy.assert_not_called()
        
        # Results should be equal
        assert result1 == result2

    @patch('src.classification.classifier.query_perplexity')
    @patch('src.classification.classifier.rule_based_classify')
    def test_cache_miss_calls_function(self, mock_rule_based, mock_llm):
        """Test that cache misses call the actual function."""
        mock_rule_based.return_value = {
            "is_chain": False,
            "confidence": 0.7,
            "reason": "Test reason",
            "method": "rule_based"
        }
        
        # First call - should call the function
        result = classify_pharmacy(SAMPLE_PHARMACY, use_llm=False)
        
        # Verify function was called
        mock_rule_based.assert_called_once()
        assert result["confidence"] == 0.7
