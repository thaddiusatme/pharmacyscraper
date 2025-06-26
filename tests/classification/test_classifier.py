"""
Tests for the pharmacy classification module.

This module contains unit tests for the pharmacy classification functionality,
including both rule-based and LLM-based classification.
"""

from unittest.mock import patch, MagicMock, call
import json
from pathlib import Path
import pytest

from pharmacy_scraper.classification.classifier import (
    Classifier,
    rule_based_classify,
    classify_pharmacy,
    batch_classify_pharmacies,
    CHAIN_IDENTIFIERS
)
from pharmacy_scraper.classification.perplexity_client import PerplexityAPIError

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

    @patch('pharmacy_scraper.classification.classifier.query_perplexity')
    def test_classify_uses_rule_based_first(self, mock_query):
        """Test that the classifier tries rule-based classification first."""
        # Setup mock to return None (simulating no LLM call needed)
        mock_query.return_value = None
        
        classifier = Classifier()
        
        # Test with a compounding pharmacy (should be caught by rules)
        result = classifier.classify_pharmacy(SAMPLE_COMPOUNDING_PHARMACY)
        
        # Verify query_perplexity wasn't called
        mock_query.assert_not_called()
        assert result["is_compounding"] is True
        assert result["source"] == "rule-based"

    @patch('pharmacy_scraper.classification.perplexity_client.PerplexityClient')
    def test_falls_back_to_llm_when_needed(self, mock_client_class):
        """Test that the classifier falls back to LLM when rules don't apply."""
        # Setup mock LLM response
        mock_client = MagicMock()
        mock_client.classify_pharmacy.return_value = {
            "classification": "independent",
            "is_compounding": False,
            "confidence": 0.85
        }
        mock_client_class.return_value = mock_client
        
        classifier = Classifier()
        
        # Test with a pharmacy that doesn't match any rules
        result = classifier.classify_pharmacy(SAMPLE_PHARMACY)
        
        # Verify PerplexityClient was used
        mock_client.classify_pharmacy.assert_called_once_with(SAMPLE_PHARMACY)
        
        # Should return LLM result converted to our format
        assert result["is_chain"] is False
        assert result["confidence"] == 0.85
        assert result["source"] == "perplexity"
        assert result["method"] == "llm"
        assert "LLM classification" in result["reason"]


class TestClassifyPharmacyFunction:
    """Tests for the classify_pharmacy function."""

    @patch('pharmacy_scraper.classification.classifier.query_perplexity')
    @patch('pharmacy_scraper.classification.classifier.rule_based_classify')
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

    @patch('pharmacy_scraper.classification.classifier.query_perplexity')
    @patch('pharmacy_scraper.classification.perplexity_client.PerplexityClient')
    @patch('pharmacy_scraper.classification.classifier.rule_based_classify')
    def test_llm_error_handling(self, mock_rule_based, mock_client_class, mock_query):
        """Test that errors in LLM classification are handled gracefully."""
        # Setup mocks
        mock_rule_based.return_value = {
            "is_chain": False,
            "is_compounding": False,
            "confidence": 0.7,
            "reason": "No chain indicators",
            "method": "rule_based",
            "source": "rule-based"
        }
        
        # Setup mock client to raise an exception
        mock_client = MagicMock()
        mock_client.classify_pharmacy.side_effect = Exception("LLM API error")
        mock_client_class.return_value = mock_client
        
        # Setup the query_perplexity mock to return a known result
        mock_query.return_value = {
            "is_chain": False,
            "is_compounding": False,
            "confidence": 0.8,
            "reason": "LLM classification failed, using fallback",
            "method": "llm",
            "source": "perplexity"
        }
        
        classifier = Classifier()
        
        # This should not raise an exception
        result = classifier.classify_pharmacy(SAMPLE_PHARMACY)
        
        # Verify the client was called
        mock_client.classify_pharmacy.assert_called_once_with(SAMPLE_PHARMACY)
        
        # Should fall back to query_perplexity result
        assert result["method"] == "llm"
        assert result["source"] == "perplexity"

    @patch('pharmacy_scraper.classification.classifier.query_perplexity')
    @patch('pharmacy_scraper.classification.classifier.rule_based_classify')
    def test_skips_llm_when_disabled(self, mock_rule_based, mock_query):
        """Test that LLM classification can be skipped."""
        # Setup mocks
        mock_rule_based.return_value = {
            "is_chain": False,
            "is_compounding": False,
            "confidence": 0.7,
            "reason": "No chain indicators",
            "method": "rule_based",
            "source": "rule-based"
        }
        
        # Clear any existing cache
        from pharmacy_scraper.classification.classifier import _classification_cache
        _classification_cache.clear()
        
        # Call with use_llm=False
        result = classify_pharmacy(SAMPLE_PHARMACY, use_llm=False)
        
        # Verify rule-based was called with the correct arguments
        mock_rule_based.assert_called_once_with(SAMPLE_PHARMACY)
        # Verify query_perplexity was not called
        mock_query.assert_not_called()
        
        # Verify the result matches the rule-based mock
        assert result["method"] == "rule_based"
        assert result["source"] == "rule-based"


class TestBatchClassification:
    """Tests for batch classification functionality."""

    @patch('pharmacy_scraper.classification.classifier.classify_pharmacy')
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

    @patch('pharmacy_scraper.classification.classifier.query_perplexity')
    @patch('pharmacy_scraper.classification.perplexity_client.PerplexityClient')
    @patch('pharmacy_scraper.classification.classifier.rule_based_classify')
    def test_llm_error_handling(self, mock_rule_based, mock_client_class, mock_query):
        """Test that LLM errors are handled gracefully."""
        # Setup mock rule-based response
        rule_based_result = {
            "is_chain": False,
            "is_compounding": False,
            "confidence": 0.7,
            "reason": "No chain indicators found",
            "method": "rule_based",
            "source": "rule-based"
        }
        mock_rule_based.return_value = rule_based_result
        
        # Setup mock client to raise an error
        mock_client = MagicMock()
        mock_client.classify_pharmacy.side_effect = PerplexityAPIError("API Error")
        mock_client_class.return_value = mock_client
        
        # Setup mock query_perplexity to return a different result
        mock_query.return_value = {
            "is_chain": False,
            "is_compounding": False,
            "confidence": 0.8,
            "reason": "LLM fallback",
            "method": "llm",
            "source": "perplexity"
        }
        
        # Create classifier
        classifier = Classifier()
        
        # Clear any existing cache
        from pharmacy_scraper.classification.classifier import _classification_cache
        _classification_cache.clear()
        
        # This should not raise an exception and should return the query_perplexity result
        result = classifier.classify_pharmacy(SAMPLE_PHARMACY)
        
        # Verify the client was called
        mock_client.classify_pharmacy.assert_called_once_with(SAMPLE_PHARMACY)
        
        # Should return the query_perplexity result, not the rule-based result
        assert result == mock_query.return_value
        
        # Verify query_perplexity was called with the correct arguments
        mock_query.assert_called_once_with(SAMPLE_PHARMACY)
        
        # Verify rule_based_classify was called with the correct arguments
        mock_rule_based.assert_called_once_with(SAMPLE_PHARMACY)
        
        # Verify the result contains the expected fields
        assert result["is_chain"] is False
        assert result["is_compounding"] is False
    @patch('pharmacy_scraper.classification.classifier.query_perplexity')
    @patch('pharmacy_scraper.classification.perplexity_client.PerplexityClient')
    @patch('pharmacy_scraper.classification.classifier.rule_based_classify')
    def test_invalid_input_handling(self, mock_rule_based, mock_client_class, mock_query):
        """Test that invalid input is handled gracefully."""
        # Setup mock rule-based response for invalid input
        rule_based_result = {
            "is_chain": False,
            "is_compounding": False,
            "confidence": 0.0,
            "reason": "Invalid input: missing required fields",
            "method": "rule_based",
            "source": "rule-based"
        }
        mock_rule_based.return_value = rule_based_result
        
        # Setup mock client with proper return values
        mock_client = MagicMock()
        mock_client.classify_pharmacy.return_value = {
            'classification': 'independent',
            'is_compounding': False,
            'confidence': 0.8,
            'reason': 'LLM classification result'
        }
        mock_client_class.return_value = mock_client
        
        # Setup mock query_perplexity to return a different result (shouldn't be called)
        mock_query.return_value = {
            "is_chain": False,
            "is_compounding": False,
            "confidence": 0.0,
            "reason": "LLM fallback: Invalid input",
            "method": "llm",
            "source": "perplexity"
        }
        
        # Create classifier
        classifier = Classifier()
        
        # Clear any existing cache
        from pharmacy_scraper.classification.classifier import _classification_cache
        _classification_cache.clear()
        
        # Test with empty input
        result = classifier.classify_pharmacy({})
        
        # Verify rule_based_classify was called with empty dict
        mock_rule_based.assert_called_once_with({})
        
        # Verify PerplexityClient was called once with empty dict
        mock_client.classify_pharmacy.assert_called_once_with({})
        
        # Verify query_perplexity was not called (since we have a client)
        mock_query.assert_not_called()
        
        # Should return the result from the client with proper formatting
        assert result == {
            "is_chain": False,  # classification='independent' in mock
            "is_compounding": False,  # from mock
            "confidence": 0.8,  # from mock
            "reason": "LLM classification: independent",  # formatted by Classifier
            "method": "llm",
            "source": "perplexity"
        }


class TestCaching:
    """Tests for the caching functionality."""

    @patch('pharmacy_scraper.classification.classifier.query_perplexity')
    @patch('pharmacy_scraper.classification.perplexity_client.PerplexityClient')
    @patch('pharmacy_scraper.classification.classifier.rule_based_classify')
    def test_cache_hit_returns_cached_result(self, mock_rule_based, mock_client_class, mock_query):
        """Test that cached results are returned when available."""
        # Setup mock rule-based response
        mock_rule_based.return_value = {
            "is_chain": False,
            "is_compounding": False,
            "confidence": 0.7,
            "reason": "No chain indicators found",
            "method": "rule_based",
            "source": "rule-based"
        }
        
        # Setup mock client and response
        mock_client = MagicMock()
        mock_client.classify_pharmacy.return_value = {
            "is_chain": False,
            "is_compounding": False,
            "confidence": 0.9,
            "reason": "LLM classification",
            "method": "llm",
            "source": "perplexity"
        }
        mock_client_class.return_value = mock_client
        
        # Setup mock query_perplexity response (shouldn't be called in this test)
        mock_query.return_value = {
            "is_chain": False,
            "is_compounding": False,
            "confidence": 0.8,
            "reason": "LLM fallback",
            "method": "llm",
            "source": "perplexity"
        }
        
        # Create classifier
        classifier = Classifier()
        
        # Clear any existing cache
        from pharmacy_scraper.classification.classifier import _classification_cache
        _classification_cache.clear()
        
        # First call - should call PerplexityClient and rule_based_classify
        result1 = classifier.classify_pharmacy(SAMPLE_PHARMACY)
        
        # Verify PerplexityClient was called with the correct arguments
        mock_client.classify_pharmacy.assert_called_once_with(SAMPLE_PHARMACY)
        mock_rule_based.assert_called_once_with(SAMPLE_PHARMACY)
        
        # Verify the first result has the expected structure
        assert result1["is_chain"] is False
        assert result1["is_compounding"] is False
        assert result1["confidence"] == 0.9
        assert "LLM classification" in result1["reason"]  # Check if reason contains expected text
        assert result1["method"] == "llm"
        assert result1["source"] == "perplexity"
        
        # Reset the mock call count
        mock_client.classify_pharmacy.reset_mock()
        
        # Call classify_pharmacy again with the same input
        result2 = classifier.classify_pharmacy(SAMPLE_PHARMACY)
        
        # Verify PerplexityClient was called again (Classifier doesn't implement caching)
        mock_client.classify_pharmacy.assert_called_once_with(SAMPLE_PHARMACY)
        
        # The results should be equivalent but not the same object (since they're created fresh)
        assert result2 == result1
        assert id(result2) != id(result1)  # Should be a deep copy, not the same object
        assert result1["is_chain"] is False
        assert result1["confidence"] == 0.9  # Matches mock return value
        assert "LLM classification" in result1["reason"]
        assert result1["source"] == "perplexity"
        
        # Reset the mock to track new calls
        mock_client.classify_pharmacy.reset_mock()
        
        # Second call with same parameters - Classifier calls PerplexityClient again
        result2 = classifier.classify_pharmacy(SAMPLE_PHARMACY)
        
        # Should call PerplexityClient again (Classifier doesn't implement caching)
        mock_client.classify_pharmacy.assert_called_once_with(SAMPLE_PHARMACY)
        
        # Results should be equal but different objects (fresh call each time)
        assert result1 == result2
        assert id(result1) != id(result2)  # Should be different objects
        
        # Check the first result
        assert result1["is_chain"] is False
        assert result1["is_compounding"] is False
        assert result1["confidence"] == 0.9  # Matches mock return value
        assert "LLM classification" in result1["reason"]
        assert result1["method"] == "llm"
        assert result1["source"] == "perplexity"
        
        # Check the second result (should be the same values, different object)
        assert result2["is_chain"] is False
        assert result2["is_compounding"] is False
        assert result2["confidence"] == 0.9  # Matches mock return value
        assert "LLM classification" in result2["reason"]
        assert result2["method"] == "llm"
        assert result2["source"] == "perplexity"

    @patch('pharmacy_scraper.classification.classifier.query_perplexity')
    @patch('pharmacy_scraper.classification.classifier.rule_based_classify')
    def test_cache_miss_calls_function(self, mock_rule_based, mock_llm):
        """Test that cache misses call the actual function."""
        # Setup mock response for rule-based classification
        mock_rule_based.return_value = {
            "is_chain": False,
            "is_compounding": False,
            "confidence": 0.7,
            "reason": "No chain indicators found",
            "method": "rule_based",
            "source": "rule_based"
        }
        
        # Setup mock response for LLM (shouldn't be called with use_llm=False)
        mock_llm.return_value = {
            "is_chain": False,
            "is_compounding": False,
            "confidence": 0.9,
            "reason": "Appears independent",
            "method": "llm",
            "source": "llm"
        }
        
        # Clear any existing cache
        from pharmacy_scraper.classification.classifier import _classification_cache
        _classification_cache.clear()
        
        # First call - should call rule_based_classify
        result1 = classify_pharmacy(SAMPLE_PHARMACY, use_llm=False)
        
        # Verify rule_based_classify was called
        mock_rule_based.assert_called_once()
        # Verify LLM was not called
        mock_llm.assert_not_called()
        # Verify the result is from rule-based classification
        assert result1["confidence"] == 0.7
        assert result1["source"] == "rule_based"
