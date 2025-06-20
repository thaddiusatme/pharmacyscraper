"""
Tests for the pharmacy classifier with caching.
"""
import os
import sys
import tempfile
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

from classification.classifier import (
    classify_pharmacy,
    batch_classify_pharmacies,
    rule_based_classify
)

# Sample test data
SAMPLE_PHARMACY = {
    "name": "Test Pharmacy",
    "address": "123 Test St, Testville",
    "phone": "555-123-4567"
}

SAMPLE_PHARMACIES = [
    {"name": "Pharmacy A", "address": "123 St", "phone": "111-1111"},
    {"name": "Pharmacy B", "address": "456 St", "phone": "222-2222"},
    {"name": "Pharmacy C", "address": "789 St", "phone": "333-3333"}
]

class TestClassifierCache:
    """Test caching functionality in the classifier."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary cache directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @patch('classification.classifier.perplexity_classify')
    def test_classify_pharmacy_caching(self, mock_classify, temp_cache_dir):
        """Test that classify_pharmacy uses the cache for repeated requests."""
        # Setup mock response
        mock_response = {
            "is_chain": False,
            "confidence": 0.95,
            "reason": "Test response"
        }
        mock_classify.return_value = mock_response
        
        # First call - should call the API
        result1 = classify_pharmacy(SAMPLE_PHARMACY, cache_dir=temp_cache_dir)
        
        # Second call with same data - should use cache
        result2 = classify_pharmacy(SAMPLE_PHARMACY, cache_dir=temp_cache_dir)
        
        # Verify mock was only called once
        mock_classify.assert_called_once()
        
        # Verify results are the same
        assert result1 == result2
        assert result1["is_chain"] == mock_response["is_chain"]
        assert result1["confidence"] == mock_response["confidence"]
        assert result1["reason"] == mock_response["reason"]
    
    @patch('classification.classifier.perplexity_classify')
    def test_batch_classify_pharmacies_caching(self, mock_classify, temp_cache_dir):
        """Test that batch_classify_pharmacies uses the cache."""
        # Setup mock response to match rule-based fallback confidence
        mock_response = {
            "is_chain": False,
            "confidence": 0.7,  
            "reason": "Test response",
            "method": "llm"
        }
        mock_classify.return_value = mock_response
        
        # Create test dataframe
        df = pd.DataFrame(SAMPLE_PHARMACIES)
        
        # First batch - should call API for each unique pharmacy
        results1 = batch_classify_pharmacies(
            df, 
            cache_dir=temp_cache_dir,
            batch_size=2
        )
        
        # Second batch with same data - should use cache
        results2 = batch_classify_pharmacies(
            df,
            cache_dir=temp_cache_dir,
            batch_size=2
        )
        
        # Verify mock was called exactly once per unique pharmacy
        assert mock_classify.call_count == len(SAMPLE_PHARMACIES)
        
        # Verify all results have the expected structure
        for result in results1.to_dict('records'):
            assert 'is_chain' in result
            assert 'confidence' in result
            assert 'classification_method' in result
            assert 'classification_reason' in result
        
        # Verify results are the same between batches
        pd.testing.assert_frame_equal(results1, results2)
    
    @patch('classification.classifier.perplexity_classify')
    def test_cache_persistence(self, mock_classify, temp_cache_dir):
        """Test that cache persists between classifier instances."""
        # Setup mock response
        mock_response = {
            "is_chain": True,
            "confidence": 0.98,
            "reason": "Test response",
            "method": "llm"
        }
        mock_classify.return_value = mock_response
        
        # First call - should call the API
        result1 = classify_pharmacy(SAMPLE_PHARMACY, cache_dir=temp_cache_dir)
        
        # Reset mock to ensure it's not called again
        mock_classify.reset_mock()
        
        # Second call with same data but new mock - should use cache
        with patch('classification.classifier.perplexity_classify') as new_mock:
            result2 = classify_pharmacy(SAMPLE_PHARMACY, cache_dir=temp_cache_dir)
            new_mock.assert_not_called()
        
        # Verify results are the same
        assert result1 == result2

    @patch('classification.classifier.perplexity_classify')
    def test_cache_key_uniqueness(self, mock_classify, temp_cache_dir):
        """Test that different pharmacy data generates different cache keys."""
        # Setup mock response
        mock_response = {
            "is_chain": False,
            "confidence": 0.95,
            "reason": "Test response",
            "method": "llm"
        }
        mock_classify.return_value = mock_response
        
        # Call with first set of data
        pharm1 = {"name": "Pharm A", "address": "123 St"}
        result1 = classify_pharmacy(pharm1, cache_dir=temp_cache_dir)
        
        # Call with slightly different data
        pharm2 = {"name": "Pharm A", "address": "124 St"}  # Different address
        result2 = classify_pharmacy(pharm2, cache_dir=temp_cache_dir)
        
        # Should have been called twice (different cache keys)
        assert mock_classify.call_count == 2
        assert result1 == result2  # Same mock response, but different cache entries

class TestClassifierIntegration:
    """Integration tests for the classifier with actual cache."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary cache directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    def test_rule_based_fallback(self, temp_cache_dir):
        """Test that rule-based classification works as a fallback."""
        # Create a pharmacy that will be classified by rules
        pharmacy_data = {
            "name": "Compounding Specialty Pharmacy",
            "address": "123 Main St"
        }
        
        # Should use rule-based classification (compounding in name)
        result = classify_pharmacy(pharmacy_data, cache_dir=temp_cache_dir)
        
        assert result["is_chain"] is False
        assert "compounding" in result["reason"].lower()
        assert result["method"] == "rule_based"
