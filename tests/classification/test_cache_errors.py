"""Tests for cache error handling in PerplexityClient."""
import os
import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from pharmacy_scraper.classification.perplexity_client import PerplexityClient, RateLimitError

# Sample data for testing
SAMPLE_PHARMACY = {
    "title": "Test Pharmacy",
    "address": "123 Test St, Test City, TS 12345",
    "phone": "(555) 123-4567"
}

@patch('pathlib.Path.mkdir')
def test_cache_read_permission_error(mock_mkdir):
    """Test handling of permission errors during cache read."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a read-only directory to trigger permission error
        cache_dir = Path(temp_dir) / "read_only_dir"
        
        # Mock mkdir to raise PermissionError
        mock_mkdir.side_effect = PermissionError("Permission denied")
        
        # Initialize client with the read-only cache directory
        client = PerplexityClient(
            api_key="test_key",
            cache_dir=str(cache_dir)
        )
        
        # Mock the API call to return a successful response
        with patch.object(client, '_call_api') as mock_call_api:
            mock_call_api.return_value = {"classification": "independent", "confidence": 0.9}
            
            # This should not raise an exception despite the read-only cache
            result = client.classify_pharmacy(SAMPLE_PHARMACY)
            
            # Verify the API was called (cache miss due to read error)
            assert mock_call_api.called
            assert result["classification"] == "independent"

@patch('builtins.open')
def test_cache_write_permission_error(mock_open, caplog):
    """Test handling of permission errors during cache write."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup mock to raise PermissionError on file write
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_open.return_value = mock_file
        mock_file.write.side_effect = PermissionError("Permission denied")
        
        # Initialize client with a valid cache directory
        client = PerplexityClient(
            api_key="test_key",
            cache_dir=temp_dir
        )
        
        # Mock the API call to return a successful response
        with patch.object(client, '_call_api') as mock_call_api:
            mock_call_api.return_value = {"classification": "independent", "confidence": 0.9}
            
            # This should not raise an exception despite the write error
            result = client.classify_pharmacy(SAMPLE_PHARMACY)
            
            # Verify the API was called (cache miss due to read error)
            assert mock_call_api.called
            assert result["classification"] == "independent"
            
            # Verify appropriate error was logged
            assert "Failed to write to cache file" in caplog.text

def test_cache_corrupted_file():
    """Test handling of corrupted cache files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a corrupted cache file
        cache_file = Path(temp_dir) / "corrupted.json"
        cache_file.write_text("this is not valid json")
        
        # Initialize client with the cache directory
        client = PerplexityClient(
            api_key="test_key",
            cache_dir=temp_dir
        )
        
        # Mock the API call to return a successful response
        with patch.object(client, '_call_api') as mock_call_api:
            mock_call_api.return_value = {"classification": "independent", "confidence": 0.9}
            
            # This should not raise an exception despite the corrupted cache
            result = client.classify_pharmacy(SAMPLE_PHARMACY)
            
            # Verify the API was called (cache miss due to corrupted file)
            assert mock_call_api.called
            assert result["classification"] == "independent"

def test_cache_disk_full(caplog):
    """Test handling of disk full errors during cache write."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize client with the temp directory
        client = PerplexityClient(
            api_key="test_key",
            cache_dir=temp_dir
        )
        
        # Mock the API call to return a successful response
        with patch.object(client, '_call_api') as mock_call_api, \
             patch('builtins.open', side_effect=OSError("No space left on device")) as mock_file:
            
            mock_call_api.return_value = {"classification": "independent", "confidence": 0.9}
            
            # This should not raise an exception despite the disk full error
            result = client.classify_pharmacy(SAMPLE_PHARMACY)
            
            # Verify the API was called (cache miss due to write error)
            assert mock_call_api.called
            assert result["classification"] == "independent"
            
            # Verify appropriate error was logged
            assert "Failed to write to cache file" in caplog.text

def test_cache_metrics(caplog):
    """Test that cache metrics are properly tracked."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize client with metrics enabled
        client = PerplexityClient(
            api_key="test_key",
            cache_dir=temp_dir,
            enable_metrics=True
        )
        
        # Mock the API call
        with patch.object(client, '_call_api') as mock_call_api, \
             patch.object(client, '_update_cache_metrics') as mock_update_metrics:
            
            mock_call_api.return_value = {"classification": "independent", "confidence": 0.9}
            
            # First call - should be a cache miss
            result1 = client.classify_pharmacy(SAMPLE_PHARMACY)
            
            # Second call with same data - should be a cache hit
            result2 = client.classify_pharmacy(SAMPLE_PHARMACY)
            
            # Verify metrics were updated correctly
            assert client._cache_metrics['hits'] == 1
            assert client._cache_metrics['misses'] == 1
            
            # Verify update_metrics was called
            assert mock_update_metrics.called

if __name__ == "__main__":
    pytest.main(["-v", __file__])
