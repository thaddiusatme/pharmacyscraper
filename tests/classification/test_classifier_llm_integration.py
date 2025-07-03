"""
Tests for the Classifier class and its LLM integration.

This test suite focuses on:
1. LLM fallback behavior in the Classifier class
2. Confidence comparison between rule-based and LLM results
3. Error handling during LLM classification
4. Cache behavior with LLM results
"""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, Any, Optional

from pharmacy_scraper.classification.classifier import (
    Classifier,
    _classification_cache,
    query_perplexity
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
        MagicMock: A mocked PerplexityClient instance with predefined behavior
    """
    mock_client = MagicMock(spec=PerplexityClient)
    mock_client.model_name = "test-model"
    mock_client.classify_pharmacy.return_value = ClassificationResult(
        is_chain=False,
        is_compounding=False,
        confidence=0.85,
        explanation="LLM classification result",
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


class TestClassifierLLMIntegration:
    """Tests for the Classifier class LLM integration."""
    
    def test_llm_fallback_with_low_confidence(self, mock_perplexity_client: MagicMock) -> None:
        """Test that LLM is used when rule-based has low confidence.
        
        Args:
            mock_perplexity_client: Mock of the PerplexityClient
        """
        # Create classifier with mock client
        classifier = Classifier(client=mock_perplexity_client)
        
        # Create test pharmacy that will get low confidence from rule-based
        pharmacy = PharmacyData(name="Generic Pharmacy")
        
        # Mock rule_based_classify to return low confidence
        with patch('pharmacy_scraper.classification.classifier.rule_based_classify') as mock_rule:
            mock_rule.return_value = ClassificationResult(
                is_chain=False,
                is_compounding=False,
                confidence=0.4,  # Low confidence
                explanation="Rule-based result with low confidence",
                source=ClassificationSource.RULE_BASED
            )
            
            # Call classify_pharmacy
            result = classifier.classify_pharmacy(pharmacy, use_llm=True)
            
            # Verify LLM was called
            mock_perplexity_client.classify_pharmacy.assert_called_once()
            
            # Result should use LLM classification since it has higher confidence
            assert result.source == ClassificationSource.PERPLEXITY
            assert result.confidence == 0.85
            assert result.model == "test-model"
    
    def test_rule_based_higher_confidence_than_llm(self, mock_perplexity_client: MagicMock) -> None:
        """Test that rule-based result is used when it has higher confidence than LLM.
        
        Args:
            mock_perplexity_client: Mock of the PerplexityClient
        """
        # Create classifier with mock client
        classifier = Classifier(client=mock_perplexity_client)
        
        # Return lower confidence from LLM
        mock_perplexity_client.classify_pharmacy.return_value = ClassificationResult(
            is_chain=False,
            is_compounding=True,  # Different classification
            confidence=0.7,  # Lower than rule-based
            explanation="LLM result with medium confidence",
            source=ClassificationSource.PERPLEXITY,
            model="test-model"
        )
        
        # Create test pharmacy
        pharmacy = PharmacyData(name="Generic Pharmacy")
        
        # Mock rule_based_classify to return high confidence
        with patch('pharmacy_scraper.classification.classifier.rule_based_classify') as mock_rule:
            mock_rule.return_value = ClassificationResult(
                is_chain=False,
                is_compounding=False,  # Different from LLM
                confidence=0.8,  # Higher than LLM
                explanation="Rule-based with higher confidence",
                source=ClassificationSource.RULE_BASED
            )
            
            # Call classify_pharmacy
            result = classifier.classify_pharmacy(pharmacy, use_llm=True)
            
            # LLM should be called
            mock_perplexity_client.classify_pharmacy.assert_called_once()
            
            # Result should use rule-based classification due to higher confidence
            assert result.source == ClassificationSource.RULE_BASED
            assert result.confidence == 0.8
            assert result.is_compounding is False  # From rule-based, not LLM
    
    def test_llm_error_handling(self, mock_perplexity_client: MagicMock) -> None:
        """Test error handling when LLM classification fails.
        
        Args:
            mock_perplexity_client: Mock of the PerplexityClient
        """
        # Create classifier with mock client
        classifier = Classifier(client=mock_perplexity_client)
        
        # Configure LLM to raise an exception
        mock_perplexity_client.classify_pharmacy.side_effect = Exception("LLM error")
        
        # Create test pharmacy
        pharmacy = PharmacyData(name="Generic Pharmacy")
        
        # Mock rule_based_classify to return medium confidence
        with patch('pharmacy_scraper.classification.classifier.rule_based_classify') as mock_rule:
            expected_result = ClassificationResult(
                is_chain=False,
                is_compounding=False,
                confidence=0.6,
                explanation="Rule-based result",
                source=ClassificationSource.RULE_BASED
            )
            mock_rule.return_value = expected_result
            
            # Call classify_pharmacy - should not raise exception despite LLM error
            result = classifier.classify_pharmacy(pharmacy, use_llm=True)
            
            # LLM should be called and fail
            mock_perplexity_client.classify_pharmacy.assert_called_once()
            
            # Result should fall back to rule-based classification
            assert result.source == ClassificationSource.RULE_BASED
            assert result.confidence == 0.6
    
    def test_llm_disabled(self, mock_perplexity_client: MagicMock) -> None:
        """Test that LLM is not used when use_llm=False.
        
        Args:
            mock_perplexity_client: Mock of the PerplexityClient
        """
        # Create classifier with mock client
        classifier = Classifier(client=mock_perplexity_client)
        
        # Create test pharmacy
        pharmacy = PharmacyData(name="Generic Pharmacy")
        
        # Mock rule_based_classify to return low confidence
        with patch('pharmacy_scraper.classification.classifier.rule_based_classify') as mock_rule:
            mock_rule.return_value = ClassificationResult(
                is_chain=False,
                is_compounding=False,
                confidence=0.3,  # Very low confidence
                explanation="Rule-based with low confidence",
                source=ClassificationSource.RULE_BASED
            )
            
            # Call classify_pharmacy with use_llm=False
            result = classifier.classify_pharmacy(pharmacy, use_llm=False)
            
            # LLM should NOT be called despite low confidence
            mock_perplexity_client.classify_pharmacy.assert_not_called()
            
            # Result should be from rule-based despite low confidence
            assert result.source == ClassificationSource.RULE_BASED
            assert result.confidence == 0.3
    
    def test_query_perplexity_stub(self) -> None:
        """Test the query_perplexity stub function."""
        # This function is meant to be mocked, but should return a valid result
        pharmacy = PharmacyData(name="Test Pharmacy")
        
        result = query_perplexity(pharmacy)
        
        # Should return a default ClassificationResult
        assert isinstance(result, ClassificationResult)
        assert result.source == ClassificationSource.PERPLEXITY
        assert result.confidence == 0.75
        assert "Stub" in result.explanation
        
    def test_cache_behavior_with_llm(self, mock_perplexity_client: MagicMock) -> None:
        """Test caching behavior with LLM results.
        
        Args:
            mock_perplexity_client: Mock of the PerplexityClient
        """
        # Create classifier with mock client
        classifier = Classifier(client=mock_perplexity_client)
        
        # Create test pharmacy
        pharmacy = PharmacyData(name="Generic Pharmacy")
        
        # Mock rule_based_classify to return low confidence to trigger LLM
        with patch('pharmacy_scraper.classification.classifier.rule_based_classify') as mock_rule:
            mock_rule.return_value = ClassificationResult(
                is_chain=False,
                is_compounding=False,
                confidence=0.4,  # Low confidence
                explanation="Rule-based with low confidence",
                source=ClassificationSource.RULE_BASED
            )
            
            # First call should use LLM
            result1 = classifier.classify_pharmacy(pharmacy, use_llm=True)
            assert result1.source == ClassificationSource.PERPLEXITY
            assert mock_perplexity_client.classify_pharmacy.call_count == 1
            
            # Reset the mock to check second call
            mock_perplexity_client.classify_pharmacy.reset_mock()
            
            # Second call should use cache
            result2 = classifier.classify_pharmacy(pharmacy, use_llm=True)
            assert result2.source == ClassificationSource.CACHE
            assert "Cached" in result2.explanation
            assert mock_perplexity_client.classify_pharmacy.call_count == 0


if __name__ == "__main__":
    pytest.main()
