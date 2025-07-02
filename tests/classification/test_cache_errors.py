"""Tests for cache error handling in PerplexityClient."""
import os
import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from pharmacy_scraper.classification.perplexity_client import (
    PerplexityClient, 
    PerplexityAPIError,
    RateLimitError,
    _generate_cache_key
)
from pharmacy_scraper.classification.models import PharmacyData, ClassificationResult, ClassificationSource
from pharmacy_scraper.utils.cache import Cache

# Sample data for testing
SAMPLE_PHARMACY_DATA = PharmacyData(
    name="Test Pharmacy",
    address="123 Test St, Test City, TS 12345",
    phone="(555) 123-4567"
)

# Sample API response - using the updated format with explanation instead of reason
SAMPLE_RESPONSE = MagicMock()
SAMPLE_RESPONSE.choices = [MagicMock()]
SAMPLE_RESPONSE.choices[0].message = MagicMock()
SAMPLE_RESPONSE.choices[0].message.content = '{"classification": "independent", "is_chain": false, "is_compounding": false, "confidence": 0.9, "explanation": "Test explanation"}'

@patch('pathlib.Path.mkdir')
def test_cache_read_permission_error(mock_mkdir):
    """Test that initializing with an unwritable cache directory raises an error."""
    # Mock mkdir to raise PermissionError
    mock_mkdir.side_effect = PermissionError("Permission denied")

    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = Path(temp_dir) / "unwritable_dir"

        # Verify that initializing the client raises a PermissionError
        with pytest.raises(PermissionError, match="Permission denied"):
            PerplexityClient(
                api_key="test_key", 
                cache_dir=str(cache_dir),
                cache_ttl_seconds=0  # Use 0 to avoid TTL calculation errors
            )

@patch('pharmacy_scraper.classification.perplexity_client.Cache')
def test_cache_write_permission_error(mock_cache_class, caplog):
    """Test handling of permission errors during cache write."""
    # Create a mock Cache instance
    mock_cache = MagicMock(spec=Cache)
    # Make get() return None to simulate a cache miss
    mock_cache.get.return_value = None
    # Make set() raise PermissionError to simulate write permission issue
    mock_cache.set.side_effect = PermissionError("Permission denied")
    # Make the mock_cache_class return our mock_cache
    mock_cache_class.return_value = mock_cache
    
    # Initialize client with cache enabled
    client = PerplexityClient(
        api_key="test_key",
        cache_dir="/tmp/test",  # This won't be used since we're mocking Cache
        cache_ttl_seconds=0  # Use 0 to avoid TTL calculation errors
    )
    
    # Mock the API call to return a successful response
    with patch.object(client, '_call_api_with_retries') as mock_call_api:
        mock_call_api.return_value = SAMPLE_RESPONSE
        
        # This should not raise an exception despite the cache write error
        result = client.classify_pharmacy(SAMPLE_PHARMACY_DATA)
        
        # Verify the API was called
        assert mock_call_api.called
        # Verify we got the expected result
        assert result.classification == "independent"
        
        # Verify our mock was called correctly
        mock_cache.get.assert_called_once()
        mock_cache.set.assert_called_once()
        
        # Verify appropriate error was logged
        assert "Cache write error" in caplog.text

@patch('pharmacy_scraper.classification.perplexity_client.Cache')
def test_cache_corrupted_file(mock_cache_class):
    """Test handling of corrupted cache files."""
    # Create a mock Cache instance
    mock_cache = MagicMock(spec=Cache)
    # Make get() raise JSONDecodeError to simulate corrupted cache
    mock_cache.get.side_effect = json.JSONDecodeError("Corrupted JSON", "content", 0)
    # Make the mock_cache_class return our mock_cache
    mock_cache_class.return_value = mock_cache
    
    # Initialize client with cache enabled
    client = PerplexityClient(
        api_key="test_key",
        cache_dir="/tmp/test",  # This won't be used since we're mocking Cache
        cache_ttl_seconds=0  # Use 0 to avoid TTL calculation errors
    )
    
    # Mock the API call to return a successful response
    with patch.object(client, '_call_api_with_retries') as mock_call_api:
        mock_call_api.return_value = SAMPLE_RESPONSE
        
        # This should not raise an exception despite the corrupted cache
        result = client.classify_pharmacy(SAMPLE_PHARMACY_DATA)
        
        # Verify the API was called (cache miss due to corrupted file)
        assert mock_call_api.called
        assert result.classification == "independent"

@patch('pharmacy_scraper.classification.perplexity_client.Cache')
def test_cache_disk_full(mock_cache_class, caplog):
    """Test handling of disk full errors during cache write."""
    # Create a mock Cache instance
    mock_cache = MagicMock(spec=Cache)
    # Make get() return None to simulate a cache miss
    mock_cache.get.return_value = None
    # Make set() raise OSError to simulate disk full
    mock_cache.set.side_effect = OSError("No space left on device")
    # Make the mock_cache_class return our mock_cache
    mock_cache_class.return_value = mock_cache
    
    # Initialize client with cache enabled
    client = PerplexityClient(
        api_key="test_key",
        cache_dir="/tmp/test",  # This won't be used since we're mocking Cache
        cache_ttl_seconds=0  # Use 0 to avoid TTL calculation errors
    )
    
    # Mock the API call to return a successful response
    with patch.object(client, '_call_api_with_retries') as mock_call_api:
        mock_call_api.return_value = SAMPLE_RESPONSE
        
        # This should not raise an exception despite the disk full error
        result = client.classify_pharmacy(SAMPLE_PHARMACY_DATA)
        
        # Verify the API was called
        assert mock_call_api.called
        assert result.classification == "independent"
        
        # Verify our mock was called correctly
        mock_cache.get.assert_called_once()
        mock_cache.set.assert_called_once()
        
        # Verify appropriate error was logged
        assert "Cache write error" in caplog.text

def test_api_error_handling():
    """Test handling of API errors during pharmacy classification."""
    client = PerplexityClient(
        api_key="test_key", 
        cache_dir=None,  # Disable cache
        cache_ttl_seconds=0  # Use 0 to avoid TTL calculation errors
    )
    
    with patch.object(client, '_call_api_with_retries') as mock_call_api:
        mock_call_api.side_effect = Exception("API connection error")
        
        # Verify that API errors are properly wrapped in PerplexityAPIError
        with pytest.raises(PerplexityAPIError):
            client.classify_pharmacy(SAMPLE_PHARMACY_DATA)

def test_rate_limit_error():
    """Test handling of rate limit errors during API calls."""
    client = PerplexityClient(
        api_key="test_key", 
        cache_dir=None,  # Disable cache
        cache_ttl_seconds=0  # Use 0 to avoid TTL calculation errors
    )
    
    with patch.object(client, '_call_api_with_retries') as mock_call_api:
        # Create a mock response with rate limit error information
        mock_call_api.side_effect = RateLimitError("Rate limit exceeded")
        
        # Verify that rate limit errors are properly propagated
        with pytest.raises(PerplexityAPIError) as excinfo:
            client.classify_pharmacy(SAMPLE_PHARMACY_DATA)
        
        # Verify error message contains rate limit information
        assert "Rate limit" in str(excinfo.value)
