"""
Tests for the classify_pharmacies_batch method in PerplexityClient.
"""
import pytest
from unittest.mock import patch, MagicMock
from typing import List, Dict, Any, Optional

from pharmacy_scraper.classification.perplexity_client import PerplexityClient, PerplexityAPIError
from pharmacy_scraper.classification.models import ClassificationResult

class TestClassifyPharmaciesBatch:
    """Test cases for the classify_pharmacies_batch method."""
    
    @pytest.fixture
    def client(self):
        """Create a test client with a mock API key."""
        return PerplexityClient(api_key="test_key")
    
    @pytest.fixture
    def mock_pharmacies(self) -> List[Dict[str, Any]]:
        """Create a list of mock pharmacy data for testing."""
        return [
            {
                "title": "Test Pharmacy 1",
                "address": "123 Test St, Testville, TS 12345",
                "categoryName": "Pharmacy, Health",
                "website": "testpharmacy1.com"
            },
            {
                "title": "Test Pharmacy 2",
                "address": "456 Example Ave, Sampletown, ST 67890",
                "categoryName": "Drugstore, Pharmacy",
                "website": "testpharmacy2.com"
            }
        ]
    
    def test_classify_pharmacies_batch_success(
        self, 
        client: PerplexityClient, 
        mock_pharmacies: List[Dict[str, Any]]
    ):
        """Test successful batch classification of pharmacies."""
        # Mock the classify_pharmacy method to return a mock result
        mock_result = {
            "is_chain": False,
            "is_compounding": False,
            "confidence": 0.92,
            "reason": "Test reason",
            "method": "llm",
            "source": "perplexity"
        }
        
        with patch.object(client, 'classify_pharmacy', return_value=mock_result) as mock_classify:
            # Call the method
            results = client.classify_pharmacies_batch(mock_pharmacies)
            
            # Verify results
            assert len(results) == 2
            assert all(isinstance(result, dict) for result in results)
            assert results[0] == mock_result
            assert results[1] == mock_result
            
            # Verify classify_pharmacy was called with each pharmacy
            assert mock_classify.call_count == 2
            assert mock_classify.call_args_list[0][0][0] == mock_pharmacies[0]
            assert mock_classify.call_args_list[1][0][0] == mock_pharmacies[1]
    
    def test_classify_pharmacies_batch_with_none_results(
        self, 
        client: PerplexityClient, 
        mock_pharmacies: List[Dict[str, Any]]
    ):
        """Test batch classification when some classifications return None."""
        # Mock the classify_pharmacy method to return None for one pharmacy
        mock_result = {
            "is_chain": False,
            "is_compounding": False,
            "confidence": 0.92,
            "reason": "Test reason",
            "method": "llm",
            "source": "perplexity"
        }
        
        with patch.object(client, 'classify_pharmacy', side_effect=[mock_result, None]) as mock_classify:
            # Call the method
            results = client.classify_pharmacies_batch(mock_pharmacies)
            
            # Verify results
            assert len(results) == 2
            assert results[0] == mock_result
            assert results[1] is None
    
    def test_classify_pharmacies_batch_empty_input(
        self, 
        client: PerplexityClient
    ):
        """Test that empty input returns empty list."""
        results = client.classify_pharmacies_batch([])
        assert results == []
    
    def test_classify_pharmacies_batch_with_exception(
        self, 
        client: PerplexityClient, 
        mock_pharmacies: List[Dict[str, Any]]
    ):
        """Test that exceptions in classify_pharmacy are caught and None is returned."""
        # Mock the classify_pharmacy method to return None (which is what happens when an exception is caught)
        with patch.object(client, 'classify_pharmacy', return_value=None) as mock_classify:
            # Call the method
            results = client.classify_pharmacies_batch(mock_pharmacies)
            
            # Verify results
            assert len(results) == 2
            assert results[0] is None
            assert results[1] is None
            
            # Verify classify_pharmacy was called with each pharmacy
            assert mock_classify.call_count == 2
    
    def test_classify_pharmacies_batch_with_model_override(
        self, 
        client: PerplexityClient, 
        mock_pharmacies: List[Dict[str, Any]]
    ):
        """Test that model parameter is passed to classify_pharmacy."""
        # Mock the classify_pharmacy method
        mock_result = {
            "is_chain": False,
            "is_compounding": False,
            "confidence": 0.92,
            "reason": "Test reason",
            "method": "llm",
            "source": "perplexity"
        }
        
        # Create a side effect function to capture the model parameter
        def side_effect(pharmacy_data, model=None):
            return mock_result
            
        with patch.object(client, 'classify_pharmacy', side_effect=side_effect) as mock_classify:
            # Call the method with a custom model
            model_override = "test-model"
            results = client.classify_pharmacies_batch(mock_pharmacies, model=model_override)
            
            # Verify the model parameter was passed to classify_pharmacy
            assert mock_classify.call_count == 2
            # Get the model parameter from the positional arguments (index 1)
            assert mock_classify.call_args_list[0][0][1] == model_override
            assert mock_classify.call_args_list[1][0][1] == model_override
