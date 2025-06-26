"""
Additional tests for improving coverage of the Perplexity client.

These tests focus on edge cases and error conditions that aren't covered
by the main test suite.
"""
import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from pharmacy_scraper.classification.perplexity_client import (
    PerplexityClient,
    PerplexityAPIError,
    RateLimitError,
    ResponseParseError
)

# Sample test data
SAMPLE_PHARMACY = {
    'name': 'Test Pharmacy',
    'address': '123 Main St',
    'city': 'Test City',
    'state': 'CA',
    'zip': '12345',
    'phone': '(555) 123-4567',
    'is_chain': False
}

class TestPerplexityClientCoverage:
    """Tests for improving coverage of the PerplexityClient class."""

    def test_cache_read_error(self, tmp_path):
        """Test handling of cache read errors."""
        # Setup mock to raise an error when reading from cache
        with patch('builtins.open', mock_open()) as mock_file:
            mock_file.side_effect = IOError("Failed to read cache")
            
            # Create a client with caching enabled
            client = PerplexityClient(
                api_key="test-key",
                cache_dir=tmp_path,
                force_reclassification=False
            )
            
            # Mock the _call_api method to return a successful result
            with patch.object(client, '_call_api') as mock_call_api:
                mock_call_api.return_value = {"is_chain": False, "confidence": 0.9}
                
                # This should not raise an exception
                result = client.classify_pharmacy(SAMPLE_PHARMACY)
                
                # Verify the API was called
                mock_call_api.assert_called_once()
                assert result == {"is_chain": False, "confidence": 0.9}

    def test_cache_write_error(self, tmp_path):
        """Test handling of cache write errors."""
        # Setup mock to raise an error when writing to cache
        with patch('builtins.open', mock_open()) as mock_file:
            mock_file.return_value.__enter__.side_effect = IOError("Failed to write cache")
            
            # Create a client with caching enabled
            client = PerplexityClient(
                api_key="test-key",
                cache_dir=tmp_path,
                force_reclassification=True
            )
            
            # Mock the _call_api method to return a successful result
            with patch.object(client, '_call_api') as mock_call_api:
                mock_call_api.return_value = {"is_chain": True, "confidence": 0.95}
                
                # This should not raise an exception
                result = client.classify_pharmacy(SAMPLE_PHARMACY)
                
                # Verify the API was called and result is still returned
                mock_call_api.assert_called_once()
                assert result == {"is_chain": True, "confidence": 0.95}

    def test_parse_response_invalid_json(self):
        """Test handling of invalid JSON in API response."""
        client = PerplexityClient(api_key="test-key")
        
        # Create a mock response with invalid JSON
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        
        # Return invalid JSON that can't be parsed
        mock_message.content = '{"invalid": "json"'
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        
        # This should return None due to invalid JSON
        result = client._parse_response(mock_response)
        assert result is None

    def test_parse_response_missing_fields(self):
        """Test handling of missing required fields in API response."""
        client = PerplexityClient(api_key="test-key")
        
        # Create a mock response with missing required fields
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        
        # Response is missing required fields like 'classification'
        mock_message.content = '{"is_chain": true, "confidence": 0.8}'
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        
        # This should return None due to missing required fields
        result = client._parse_response(mock_response)
        assert result is None

    def test_call_api_retry_success(self):
        """Test that API retries on transient errors."""
        client = PerplexityClient(api_key="test-key", max_retries=2)
        
        # Setup mock to fail once then succeed
        with patch.object(client.client.chat.completions, 'create') as mock_create:
            # First call fails with rate limit, second succeeds
            mock_create.side_effect = [
                Exception("rate limit"),
                MagicMock(choices=[MagicMock(message=MagicMock(content='{"classification": "independent", "is_compounding": false, "confidence": 0.9}'))])
            ]
            
            result = client._call_api("test prompt")
            assert result == {"classification": "independent", "is_compounding": False, "confidence": 0.9}
            assert mock_create.call_count == 2

    def test_call_api_rate_limit_exceeded(self):
        """Test that RateLimitError is raised when retries are exhausted."""
        client = PerplexityClient(api_key="test-key", max_retries=1)
        
        # Setup mock to always fail with rate limit
        with patch.object(client.client.chat.completions, 'create') as mock_create:
            mock_create.side_effect = Exception("rate limit")
            
            with pytest.raises(RateLimitError):
                client._call_api("test prompt")
            
            # Should have tried max_retries + 1 times (initial + retries)
            assert mock_create.call_count == 2
