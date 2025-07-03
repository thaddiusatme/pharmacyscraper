"""
Tests for batch classification functionality in the classifier module.

This test suite focuses on:
1. Batch processing of multiple pharmacies
2. Error handling in batch mode
3. Edge cases in batch processing
4. Parameter passing from batch to individual methods
"""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, List, Optional, Union, Any

from pharmacy_scraper.classification.classifier import (
    Classifier,
    _classification_cache
)
from pharmacy_scraper.classification.models import (
    PharmacyData,
    ClassificationResult,
    ClassificationSource
)
from pharmacy_scraper.classification.perplexity_client import PerplexityClient


@pytest.fixture
def mock_perplexity_client() -> MagicMock:
    """Create a mock PerplexityClient for testing.
    
    Returns:
        MagicMock: A mocked PerplexityClient
    """
    mock_client = MagicMock(spec=PerplexityClient)
    mock_client.model_name = "test-model"
    mock_client.classify_pharmacy.return_value = ClassificationResult(
        is_chain=False,
        is_compounding=False,
        confidence=0.8,
        explanation="LLM result",
        source=ClassificationSource.PERPLEXITY,
        model="test-model"
    )
    return mock_client


@pytest.fixture(autouse=True)
def clear_cache() -> None:
    """Clear the classification cache before and after each test."""
    _classification_cache.clear()
    yield
    _classification_cache.clear()


class TestBatchClassification:
    """Tests for batch classification functionality."""
    
    def test_batch_with_multiple_pharmacies(self, mock_perplexity_client: MagicMock) -> None:
        """Test batch classification with multiple pharmacies.
        
        Args:
            mock_perplexity_client: Mock of the PerplexityClient
        """
        # Create classifier
        classifier = Classifier(client=mock_perplexity_client)
        
        # Create test pharmacies
        pharmacies = [
            PharmacyData(name="Pharmacy 1", address="Address 1"),
            PharmacyData(name="Pharmacy 2", address="Address 2"),
            PharmacyData(name="Pharmacy 3", address="Address 3")
        ]
        
        # Mock classify_pharmacy to track calls
        with patch.object(classifier, 'classify_pharmacy') as mock_classify:
            # Set return values for each call
            mock_classify.side_effect = [
                ClassificationResult(is_chain=True, is_compounding=False, confidence=0.9, source=ClassificationSource.RULE_BASED),
                ClassificationResult(is_chain=False, is_compounding=True, confidence=0.95, source=ClassificationSource.RULE_BASED),
                ClassificationResult(is_chain=False, is_compounding=False, confidence=0.7, source=ClassificationSource.PERPLEXITY)
            ]
            
            # Call batch_classify_pharmacies
            results = classifier.batch_classify_pharmacies(pharmacies)
            
            # Verify classify_pharmacy was called for each pharmacy
            assert mock_classify.call_count == 3
            assert len(results) == 3
            
            # Verify each result is a ClassificationResult
            for result in results:
                assert isinstance(result, ClassificationResult)
    
    def test_batch_with_exception_handling(self, mock_perplexity_client: MagicMock) -> None:
        """Test batch classification handles exceptions for individual pharmacies.
        
        Args:
            mock_perplexity_client: Mock of the PerplexityClient
        """
        # Create classifier
        classifier = Classifier(client=mock_perplexity_client)
        
        # Create test pharmacies
        pharmacies = [
            PharmacyData(name="Good Pharmacy"),
            PharmacyData(name="Error Pharmacy"),
            PharmacyData(name="Another Good Pharmacy")
        ]
        
        # Mock classify_pharmacy to raise exception for the second pharmacy
        def classify_side_effect(pharmacy: Union[Dict, PharmacyData], **kwargs: Any) -> ClassificationResult:
            if pharmacy.name == "Error Pharmacy":
                raise ValueError("Test error")
            return ClassificationResult(
                is_chain=False,
                is_compounding=False,
                confidence=0.8,
                source=ClassificationSource.RULE_BASED
            )
        
        with patch.object(classifier, 'classify_pharmacy') as mock_classify:
            mock_classify.side_effect = classify_side_effect
            
            # Call batch_classify_pharmacies
            results = classifier.batch_classify_pharmacies(pharmacies)
            
            # Verify all pharmacies were processed
            assert mock_classify.call_count == 3
            assert len(results) == 3
            
            # First and third results should be ClassificationResults, second should be None
            assert isinstance(results[0], ClassificationResult)
            assert results[1] is None
            assert isinstance(results[2], ClassificationResult)
    
    def test_batch_with_mixed_input_types(self, mock_perplexity_client: MagicMock) -> None:
        """Test batch classification with mixed input types (dict and PharmacyData).
        
        Args:
            mock_perplexity_client: Mock of the PerplexityClient
        """
        # Create classifier
        classifier = Classifier(client=mock_perplexity_client)
        
        # Create mixed test data
        pharmacy_dict = {"name": "Dict Pharmacy", "address": "123 Dict St"}
        pharmacy_data = PharmacyData(name="Data Pharmacy", address="456 Data Ave")
        pharmacies = [pharmacy_dict, pharmacy_data]
        
        # Mock classify_pharmacy to track calls
        with patch.object(classifier, 'classify_pharmacy') as mock_classify:
            # Set return values
            mock_classify.return_value = ClassificationResult(
                is_chain=False,
                is_compounding=False,
                confidence=0.8,
                source=ClassificationSource.RULE_BASED
            )
            
            # Call batch_classify_pharmacies
            results = classifier.batch_classify_pharmacies(pharmacies)
            
            # Verify classify_pharmacy was called for each pharmacy
            assert mock_classify.call_count == 2
            assert len(results) == 2
            
            # Check that both input types were passed correctly
            call_args_list = mock_classify.call_args_list
            assert call_args_list[0][0][0] == pharmacy_dict
            assert call_args_list[1][0][0] == pharmacy_data
    
    def test_batch_with_empty_list(self, mock_perplexity_client: MagicMock) -> None:
        """Test batch classification with empty list.
        
        Args:
            mock_perplexity_client: Mock of the PerplexityClient
        """
        # Create classifier
        classifier = Classifier(client=mock_perplexity_client)
        
        # Mock classify_pharmacy to track calls
        with patch.object(classifier, 'classify_pharmacy') as mock_classify:
            # Call batch_classify_pharmacies with empty list
            results = classifier.batch_classify_pharmacies([])
            
            # Verify no calls to classify_pharmacy
            assert mock_classify.call_count == 0
            # Should return empty list
            assert results == []
    
    def test_batch_use_llm_parameter(self, mock_perplexity_client: MagicMock) -> None:
        """Test use_llm parameter passing in batch classification.
        
        Args:
            mock_perplexity_client: Mock of the PerplexityClient
        """
        # Create classifier
        classifier = Classifier(client=mock_perplexity_client)
        
        # Create test pharmacy
        pharmacy = PharmacyData(name="Test Pharmacy")
        
        # Test with use_llm=True
        with patch.object(classifier, 'classify_pharmacy') as mock_classify:
            classifier.batch_classify_pharmacies([pharmacy], use_llm=True)
            mock_classify.assert_called_once_with(pharmacy, use_llm=True)
            
        # Test with use_llm=False
        with patch.object(classifier, 'classify_pharmacy') as mock_classify:
            classifier.batch_classify_pharmacies([pharmacy], use_llm=False)
            mock_classify.assert_called_once_with(pharmacy, use_llm=False)
    
    def test_batch_with_none_elements(self, mock_perplexity_client: MagicMock) -> None:
        """Test batch classification with None elements in the list.
        
        Args:
            mock_perplexity_client: Mock of the PerplexityClient
        """
        # Create classifier
        classifier = Classifier(client=mock_perplexity_client)
        
        # Create test pharmacies with None element
        pharmacies = [
            PharmacyData(name="Good Pharmacy"),
            None,
            PharmacyData(name="Another Good Pharmacy")
        ]
        
        # Call batch_classify_pharmacies
        results = classifier.batch_classify_pharmacies(pharmacies)
        
        # None element should result in None in output
        assert len(results) == 3
        assert isinstance(results[0], ClassificationResult)
        assert results[1] is None
        assert isinstance(results[2], ClassificationResult)


if __name__ == "__main__":
    pytest.main()
