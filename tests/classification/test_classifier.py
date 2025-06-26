"""
Tests for the pharmacy classification module.

This module contains unit tests for the pharmacy classification functionality,
including both rule-based and LLM-based classification.
"""

from unittest.mock import patch, MagicMock, call
import json
from pathlib import Path
from typing import Dict, Any, Union
import pytest

from pharmacy_scraper.classification.classifier import (
    Classifier,
    rule_based_classify,
    classify_pharmacy,
    batch_classify_pharmacies,
    CHAIN_IDENTIFIERS
)
from pharmacy_scraper.classification.data_models import (
    PharmacyData,
    ClassificationResult,
    ClassificationMethod,
    ClassificationSource
)
from pharmacy_scraper.classification.perplexity_client import PerplexityAPIError

def create_pharmacy_dict(name: str, **kwargs) -> Dict[str, Any]:
    """Helper to create a pharmacy dictionary with default values."""
    defaults = {
        "name": name,
        "address": "123 Main St, Test City, CA 12345",
        "phone": "(555) 123-4567",
        "website": "https://testpharmacy.com"
    }
    defaults.update(kwargs)
    return defaults

@pytest.fixture
def sample_pharmacy() -> Dict[str, Any]:
    """Return a sample pharmacy dictionary for testing."""
    return {
        "name": "Test Pharmacy",
        "address": "123 Main St, Test City, CA 12345",
        "phone": "(555) 123-4567",
        "website": "https://testpharmacy.com"
    }
    
@pytest.fixture
def sample_pharmacy_obj(sample_pharmacy: Dict[str, Any]) -> PharmacyData:
    """Return a sample PharmacyData object for testing."""
    return PharmacyData.from_dict(sample_pharmacy)

# Sample pharmacy data for testing as dictionaries
SAMPLE_PHARMACY = create_pharmacy_dict("Test Pharmacy")
SAMPLE_CHAIN_PHARMACY = create_pharmacy_dict(
    "CVS Pharmacy #1234", 
    address="456 Oak St, Test City, CA 12345",
    website="https://www.cvs.com"
)
SAMPLE_COMPOUNDING_PHARMACY = create_pharmacy_dict(
    "Test Compounding Pharmacy",
    address="789 Pine St, Test City, CA 12345",
    website="https://testcompounding.com"
)

# Sample pharmacy data as PharmacyData objects
SAMPLE_PHARMACY_OBJ = PharmacyData.from_dict(SAMPLE_PHARMACY)
SAMPLE_CHAIN_PHARMACY_OBJ = PharmacyData.from_dict(SAMPLE_CHAIN_PHARMACY)
SAMPLE_COMPOUNDING_PHARMACY_OBJ = PharmacyData.from_dict(SAMPLE_COMPOUNDING_PHARMACY)


class TestRuleBasedClassifier:
    """Tests for the rule-based classification functionality."""

    def test_identifies_chain_pharmacies(self, sample_pharmacy: Dict[str, Any]) -> None:
        """Test that chain pharmacies are correctly identified."""
        chain_pharmacy = {**sample_pharmacy, "name": "CVS Pharmacy #1234"}
        result = rule_based_classify(chain_pharmacy)
        assert result.is_chain is True
        assert result.confidence == 1.0  # Exact match for chain pharmacy
        assert "Matched chain keyword: CVS" in result.reason
        assert result.method == ClassificationMethod.RULE_BASED
        assert result.source == ClassificationSource.RULE_BASED

    def test_identifies_compounding_pharmacies(self):
        """Test that compounding pharmacies are correctly identified."""
        # Test with dict input
        result_dict = rule_based_classify(SAMPLE_COMPOUNDING_PHARMACY)
        assert result_dict.is_chain is False
        assert result_dict.is_compounding is True
        assert "Compounding pharmacy keyword detected" in result_dict.reason
        assert result_dict.confidence == 0.95  # Exact confidence for compounding
        
        # Test with PharmacyData input
        result_obj = rule_based_classify(SAMPLE_COMPOUNDING_PHARMACY_OBJ)
        assert result_obj.is_chain is False
        assert result_obj.is_compounding is True
        assert "Compounding pharmacy keyword detected" in result_obj.reason
        assert result_obj.confidence == 0.95  # Exact confidence for compounding

    def test_defaults_to_independent(self, sample_pharmacy: Dict[str, Any]) -> None:
        """Test that pharmacies without clear indicators are classified as independent."""
        result = rule_based_classify(sample_pharmacy)
        assert result.is_chain is False
        assert result.confidence == 0.5  # Default independent confidence
        assert "No chain identifiers found" in result.reason
        assert result.method == ClassificationMethod.RULE_BASED
        assert result.source == ClassificationSource.RULE_BASED


class TestClassifierIntegration:
    """Integration tests for the main Classifier class."""

    @patch('pharmacy_scraper.classification.classifier.query_perplexity')
    def test_classify_uses_rule_based_first(self, mock_query):
        """Test that the classifier tries rule-based classification first."""
        # Setup mock to return a ClassificationResult (simulating no LLM call needed)
        mock_query.return_value = ClassificationResult(
            is_chain=False,
            is_compounding=True,
            confidence=0.9,
            reason="Compounding pharmacy detected",
            method=ClassificationMethod.RULE_BASED,
            source=ClassificationSource.RULE_BASED
        )
        
        classifier = Classifier()
        
        # Test with a compounding pharmacy (should be caught by rules)
        result = classifier.classify_pharmacy(SAMPLE_COMPOUNDING_PHARMACY)
        
        # Verify query_perplexity wasn't called
        mock_query.assert_not_called()
        assert result.is_compounding is True
        assert result.source == ClassificationSource.RULE_BASED
        
        # Also test with PharmacyData object
        result_obj = classifier.classify_pharmacy(SAMPLE_COMPOUNDING_PHARMACY_OBJ)
        assert result_obj.is_compounding is True
        assert result_obj.source == ClassificationSource.RULE_BASED

    @patch('pharmacy_scraper.classification.perplexity_client.PerplexityClient')
    def test_falls_back_to_llm_when_needed(self, mock_client_class):
        """Test that the classifier falls back to LLM when rules don't apply."""
        # Setup mock LLM response
        mock_client = MagicMock()
        mock_client.classify_pharmacy.return_value = ClassificationResult(
            is_chain=False,
            is_compounding=False,
            confidence=0.85,
            reason="Appears to be independent",
            method=ClassificationMethod.LLM,
            source=ClassificationSource.PERPLEXITY
        )
        mock_client_class.return_value = mock_client
        
        classifier = Classifier()
        
        # Test with a pharmacy that doesn't match any rules (using dict)
        result = classifier.classify_pharmacy(SAMPLE_PHARMACY)
        
        # Verify PerplexityClient was used
        mock_client.classify_pharmacy.assert_called_once()
        
        # Should return LLM result
        assert result.is_chain is False
        assert result.confidence == 0.85
        assert result.source == ClassificationSource.PERPLEXITY
        assert result.method == ClassificationMethod.LLM
        assert "Appears to be independent" in result.reason
        
        # Also test with PharmacyData object
        mock_client.reset_mock()
        result_obj = classifier.classify_pharmacy(SAMPLE_PHARMACY_OBJ)
        assert result_obj.is_chain is False
        assert result_obj.source == ClassificationSource.PERPLEXITY


class TestClassifyPharmacyFunction:
    """Tests for the classify_pharmacy function."""
    
    def setup_method(self):
        """Clear the cache before each test to ensure isolation."""
        from pharmacy_scraper.classification.classifier import _classification_cache
        _classification_cache.clear()

    @patch('pharmacy_scraper.classification.classifier.query_perplexity')
    @patch('pharmacy_scraper.classification.classifier.rule_based_classify')
    def test_uses_both_classifiers_by_default(self, mock_rule_based, mock_query):
        """Test that both classifiers are used by default."""
        # Setup mocks
        rule_based_result = ClassificationResult(
            is_chain=False,
            is_compounding=False,
            confidence=0.7,
            reason="No chain indicators",
            method=ClassificationMethod.RULE_BASED,
            source=ClassificationSource.RULE_BASED
        )
        mock_rule_based.return_value = rule_based_result
        
        llm_result = ClassificationResult(
            is_chain=False,
            is_compounding=False,
            confidence=0.8,
            reason="Appears independent",
            method=ClassificationMethod.LLM,
            source=ClassificationSource.PERPLEXITY
        )
        mock_query.return_value = llm_result
        
        # Test with dict input
        result = classify_pharmacy(SAMPLE_PHARMACY)
        
        # Verify both classifiers were called
        mock_rule_based.assert_called_once()
        mock_query.assert_called_once()
        
        # Should prefer the higher confidence result (LLM in this case)
        assert result.confidence == 0.8
        assert result.method == ClassificationMethod.LLM
        assert result.source == ClassificationSource.PERPLEXITY
        
        # Test with PharmacyData object
        mock_rule_based.reset_mock()
        mock_query.reset_mock()
        mock_rule_based.return_value = rule_based_result
        mock_query.return_value = llm_result
        
        result_obj = classify_pharmacy(SAMPLE_PHARMACY_OBJ)
        assert result_obj.confidence == 0.8
        assert result_obj.method == ClassificationMethod.LLM

    @patch('pharmacy_scraper.classification.classifier.query_perplexity')
    @patch('pharmacy_scraper.classification.perplexity_client.PerplexityClient')
    @patch('pharmacy_scraper.classification.classifier.rule_based_classify')
    def test_llm_error_handling(self, mock_rule_based, mock_client_class, mock_query):
        """Test that errors in LLM classification are handled gracefully."""
        # Setup mocks
        rule_based_result = ClassificationResult(
            is_chain=False,
            is_compounding=False,
            confidence=0.7,
            reason="No chain indicators found",
            method=ClassificationMethod.RULE_BASED,
            source=ClassificationSource.RULE_BASED
        )
        mock_rule_based.return_value = rule_based_result
        
        # Setup mock client to raise an exception
        mock_client = MagicMock()
        mock_client.classify_pharmacy.side_effect = PerplexityAPIError("API Error")
        mock_client_class.return_value = mock_client
        
        classifier = Classifier(client=mock_client)
        
        # This should not raise an exception
        result = classifier.classify_pharmacy(SAMPLE_PHARMACY)
        
        # Verify the client was called
        mock_client.classify_pharmacy.assert_called_once()
        
        # Should return the rule-based result since LLM failed
        assert result == rule_based_result
        
        # Test with PharmacyData object
        mock_client.reset_mock()
        mock_client.classify_pharmacy.side_effect = PerplexityAPIError("API Error")
        
        result_obj = classifier.classify_pharmacy(SAMPLE_PHARMACY_OBJ)
        assert result_obj == rule_based_result

    @patch('pharmacy_scraper.classification.classifier.query_perplexity')
    @patch('pharmacy_scraper.classification.classifier.rule_based_classify')
    def test_skips_llm_when_disabled(self, mock_rule_based, mock_query):
        """Test that LLM classification can be skipped."""
        # Setup mocks
        rule_based_result = ClassificationResult(
            is_chain=False,
            is_compounding=False,
            confidence=0.7,
            reason="No chain indicators",
            method=ClassificationMethod.RULE_BASED,
            source=ClassificationSource.RULE_BASED
        )
        mock_rule_based.return_value = rule_based_result
        
        # Clear any existing cache
        from pharmacy_scraper.classification.classifier import _classification_cache
        _classification_cache.clear()
        
        # Call with use_llm=False (using dict)
        result = classify_pharmacy(SAMPLE_PHARMACY, use_llm=False)
        
        # Verify rule-based was called with the correct arguments
        mock_rule_based.assert_called_once()
        # Verify query_perplexity was not called
        mock_query.assert_not_called()
        
        # Verify the result matches the rule-based mock
        assert result == rule_based_result
        assert result.method == ClassificationMethod.RULE_BASED
        assert result.source == ClassificationSource.RULE_BASED
        
        # Also test with PharmacyData object
        mock_rule_based.reset_mock()
        result_obj = classify_pharmacy(SAMPLE_PHARMACY_OBJ, use_llm=False)
        assert result_obj == rule_based_result
        assert result_obj.method == ClassificationMethod.RULE_BASED
        assert result_obj.source == ClassificationSource.RULE_BASED


class TestBatchClassification:
    """Tests for batch classification functionality."""

    @patch('pharmacy_scraper.classification.classifier.classify_pharmacy')
    def test_batch_classification(self, mock_classify):
        """Test that batch classification processes multiple pharmacies."""
        # Setup mock to return different results for different inputs
        def side_effect(pharmacy, **kwargs):
            name = pharmacy.get("name") if isinstance(pharmacy, dict) else getattr(pharmacy, "name", "")
            if name == "Pharmacy A":
                return ClassificationResult(
                    is_chain=False,
                    is_compounding=False,
                    confidence=0.9,
                    reason="Independent pharmacy",
                    method=ClassificationMethod.RULE_BASED,
                    source=ClassificationSource.RULE_BASED
                )
            return ClassificationResult(
                is_chain=True,
                is_compounding=False,
                confidence=0.95,
                reason="Chain pharmacy detected",
                method=ClassificationMethod.RULE_BASED,
                source=ClassificationSource.RULE_BASED
            )
            
        mock_classify.side_effect = side_effect
        
        # Test with list of dicts
        pharmacies = [
            {"name": "Pharmacy A", "address": "123 Test St"},
            {"name": "CVS #123", "address": "456 Test St"}
        ]
        
        results = batch_classify_pharmacies(pharmacies)
        
        assert len(results) == 2
        assert results[0].is_chain is False
        assert results[1].is_chain is True
        assert mock_classify.call_count == 2
        
        # Test with list of PharmacyData objects
        mock_classify.reset_mock()
        pharmacies_objs = [
            PharmacyData(name="Pharmacy A", address="123 Test St"),
            PharmacyData(name="CVS #123", address="456 Test St")
        ]
        
        results_objs = batch_classify_pharmacies(pharmacies_objs)
        
        assert len(results_objs) == 2
        assert results_objs[0].is_chain is False
        assert results_objs[1].is_chain is True
        assert mock_classify.call_count == 2


class TestErrorHandling:
    """Tests for error handling in the classification pipeline."""

    @patch('pharmacy_scraper.classification.classifier.query_perplexity')
    @patch('pharmacy_scraper.classification.perplexity_client.PerplexityClient')
    @patch('pharmacy_scraper.classification.classifier.rule_based_classify')
    def test_llm_error_handling(self, mock_rule_based, mock_client_class, mock_query):
        """Test that LLM errors are handled gracefully."""
        # Setup mock rule-based response
        rule_based_result = ClassificationResult(
            is_chain=False,
            is_compounding=False,
            confidence=0.7,
            reason="No chain indicators found",
            method=ClassificationMethod.RULE_BASED,
            source=ClassificationSource.RULE_BASED
        )
        mock_rule_based.return_value = rule_based_result
        
        # Setup mock client to raise an error
        mock_client = MagicMock()
        mock_client.classify_pharmacy.side_effect = PerplexityAPIError("API Error")
        mock_client_class.return_value = mock_client
        
        # Setup mock query_perplexity to return a different result
        fallback_result = ClassificationResult(
            is_chain=False,
            is_compounding=False,
            confidence=0.8,
            reason="LLM fallback",
            method=ClassificationMethod.LLM,
            source=ClassificationSource.PERPLEXITY
        )
        mock_query.return_value = fallback_result
        
        # Create classifier
        classifier = Classifier()
        
        # Clear any existing cache
        from pharmacy_scraper.classification.classifier import _classification_cache
        _classification_cache.clear()
        
        # Test with dict input
        result = classifier.classify_pharmacy(SAMPLE_PHARMACY)
        
        # Verify the client was called
        mock_client.classify_pharmacy.assert_called_once()
        
        # Should return the rule-based result when LLM fails
        assert result == rule_based_result
        
        # Test with PharmacyData object
        mock_client.reset_mock()
        mock_client.classify_pharmacy.side_effect = PerplexityAPIError("API Error")
        
        result_obj = classifier.classify_pharmacy(SAMPLE_PHARMACY_OBJ)
        assert result_obj == rule_based_result
        
        # Verify the results contain the expected fields
        assert not result.is_chain
        assert not result.is_compounding
    @patch('pharmacy_scraper.classification.classifier.query_perplexity')
    @patch('pharmacy_scraper.classification.perplexity_client.PerplexityClient')
    @patch('pharmacy_scraper.classification.classifier.rule_based_classify')
    def test_invalid_input_handling(self, mock_rule_based, mock_client_class, mock_query):
        """Test that invalid input is handled gracefully."""
        # Setup mock rule-based response for invalid input
        rule_based_result = ClassificationResult(
            is_chain=False,
            is_compounding=False,
            confidence=0.0,
            reason="Invalid input: missing required fields",
            method=ClassificationMethod.RULE_BASED,
            source=ClassificationSource.RULE_BASED
        )
        mock_rule_based.return_value = rule_based_result
        
        # Setup mock client with proper return values
        mock_client = MagicMock()
        mock_client.classify_pharmacy.return_value = ClassificationResult(
            is_chain=False,  # classification='independent'
            is_compounding=False,
            confidence=0.8,
            reason="LLM classification result",
            method=ClassificationMethod.LLM,
            source=ClassificationSource.PERPLEXITY
        )
        mock_client_class.return_value = mock_client
        
        # Setup mock query_perplexity (shouldn't be called in this test)
        fallback_result = ClassificationResult(
            is_chain=False,
            is_compounding=False,
            confidence=0.0,
            reason="LLM fallback: Invalid input",
            method=ClassificationMethod.LLM,
            source=ClassificationSource.PERPLEXITY
        )
        mock_query.return_value = fallback_result
        
        # Create classifier
        classifier = Classifier()
        
        # Clear any existing cache
        from pharmacy_scraper.classification.classifier import _classification_cache
        _classification_cache.clear()
        
        # Test with empty dict input
        result = classifier.classify_pharmacy({})
        
        # Verify rule_based_classify was called with empty dict
        mock_rule_based.assert_called_once_with({})
        
        # Verify PerplexityClient was called once with empty dict
        mock_client.classify_pharmacy.assert_called_once_with({})
        
        # Verify query_perplexity was not called (since we have a client)
        mock_query.assert_not_called()
        
        # Should return the result from the client with proper formatting
        assert result == mock_client.classify_pharmacy.return_value
        
        # Test with None input (should raise ValueError)
        mock_rule_based.reset_mock()
        mock_client.reset_mock()
        
        with pytest.raises(ValueError, match="Pharmacy data cannot be None"):
            classifier.classify_pharmacy(None)
            
        # Verify no calls were made to the mocks
        mock_rule_based.assert_not_called()
        mock_client.classify_pharmacy.assert_not_called()


class TestCaching:
    """Tests for classification caching functionality."""

    @patch('pharmacy_scraper.classification.classifier.query_perplexity')
    @patch('pharmacy_scraper.classification.perplexity_client.PerplexityClient')
    @patch('pharmacy_scraper.classification.classifier.rule_based_classify')
    def test_caching_behavior(self, mock_rule_based, mock_client_class, mock_query):
        """Test that classification results are properly cached."""
        # Setup mock rule-based response
        rule_based_result = ClassificationResult(
            is_chain=False,
            is_compounding=False,
            confidence=0.7,
            reason="No chain indicators found",
            method=ClassificationMethod.RULE_BASED,
            source=ClassificationSource.RULE_BASED
        )
        mock_rule_based.return_value = rule_based_result
        
        # Setup mock client to return a ClassificationResult with confidence 0.8
        self.mock_client = MagicMock()
        llm_result = ClassificationResult(
            is_chain=False,
            is_compounding=False,
            confidence=0.8,
            reason="LLM classification result",
            method=ClassificationMethod.LLM,
            source=ClassificationSource.PERPLEXITY
        )
        self.mock_client.classify_pharmacy.return_value = llm_result
        mock_client_class.return_value = self.mock_client
        
        # Setup mock query_perplexity to return the same result for consistency
        mock_query.return_value = llm_result
        
        # Clear cache before starting
        from pharmacy_scraper.classification.classifier import _classification_cache
        _classification_cache.clear()
        
        # Test with dict input
        classifier = Classifier()
        result1 = classifier.classify_pharmacy(SAMPLE_PHARMACY)
        
        # Verify client was called
        self.mock_client.classify_pharmacy.assert_called_once()
        
        # Reset mocks
        self.mock_client.reset_mock()
        mock_rule_based.reset_mock()
        
        # Second call with same input - should use cache
        result2 = classifier.classify_pharmacy(SAMPLE_PHARMACY)
        
        # Verify client was NOT called again
        self.mock_client.classify_pharmacy.assert_not_called()
        
        # Results should be equal
        assert result1 == result2
        
        # Verify the cache key is consistent
        from pharmacy_scraper.classification.classifier import _get_cache_key
        cache_key = _get_cache_key(SAMPLE_PHARMACY)
        assert cache_key in _classification_cache
        
        # Test with PharmacyData object - should also use cache if same data
        result_obj = classifier.classify_pharmacy(SAMPLE_PHARMACY_OBJ)
        self.mock_client.classify_pharmacy.assert_not_called()
        assert result_obj == result1
        
        # Test with different input - should call client again
        different_pharmacy = {"name": "Different Pharmacy", "address": "789 Test St"}
        result3 = classifier.classify_pharmacy(different_pharmacy)
        
        # Verify client was called with the new input
        self.mock_client.classify_pharmacy.assert_called_once()
        
        # Verify the results have the expected values
        assert result1.is_chain is False
        assert result1.is_compounding is False
        assert result1.confidence == 0.8
        assert "LLM classification" in result1.reason
        assert result1.method == ClassificationMethod.LLM
        assert result1.source == ClassificationSource.PERPLEXITY

    @patch('pharmacy_scraper.classification.classifier.query_perplexity')
    @patch('pharmacy_scraper.classification.classifier.rule_based_classify')
    def test_cache_miss_calls_function(self, mock_rule_based, mock_query):
        """Test that cache misses call the actual function."""
        # Setup mock response for rule-based classification
        rule_based_result = ClassificationResult(
            is_chain=False,
            is_compounding=False,
            confidence=0.7,
            reason="No chain indicators found",
            method=ClassificationMethod.RULE_BASED,
            source=ClassificationSource.RULE_BASED
        )
        mock_rule_based.return_value = rule_based_result
        
        # Setup mock response for LLM (shouldn't be called with use_llm=False)
        llm_result = ClassificationResult(
            is_chain=False,
            is_compounding=False,
            confidence=0.9,
            reason="Appears independent",
            method=ClassificationMethod.LLM,
            source=ClassificationSource.PERPLEXITY
        )
        mock_query.return_value = llm_result
        
        # Clear any existing cache
        from pharmacy_scraper.classification.classifier import _classification_cache
        _classification_cache.clear()
        
        # First call - should call rule_based_classify
        result = classify_pharmacy(SAMPLE_PHARMACY, use_llm=False)
        
        # Verify rule_based_classify was called
        mock_rule_based.assert_called_once_with(SAMPLE_PHARMACY)
        # Verify LLM was not called
        mock_query.assert_not_called()
        # Verify the result is from rule-based classification
        assert result == rule_based_result
        
        # Second call with use_llm=True should call query_perplexity
        mock_rule_based.reset_mock()
        result2 = classify_pharmacy(SAMPLE_PHARMACY, use_llm=True)
        
        # Verify both functions were called
        mock_rule_based.assert_called_once()
        mock_query.assert_called_once()
        # Verify the result is from LLM classification
        assert result2 == llm_result
