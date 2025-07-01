"""
Comprehensive tests for the Perplexity API client, caching, and error handling.
"""
import os
import time
import json
import pytest
import openai
import tempfile
import shutil
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

from pharmacy_scraper.classification.perplexity_client import (
    PerplexityClient,
    PerplexityAPIError,
    RateLimitError,
    ResponseParseError,
    _generate_cache_key,
    RateLimiter
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

@pytest.fixture
def mock_openai():
    """Mock the OpenAI client."""
    with patch('pharmacy_scraper.classification.perplexity_client.openai.OpenAI') as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = json.dumps({
            "classification": "independent",
            "is_compounding": False,
            "confidence": 0.9,  # Changed from 0.95 to 0.9 to be within [0, 1] range
            "explanation": "This is a mock explanation."
        })
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        
        yield mock_client

class TestPerplexityClient:
    """Comprehensive tests for the PerplexityClient class."""

    @pytest.fixture
    def client(self, mock_openai, tmp_path):
        """Create a test client with caching enabled."""
        return PerplexityClient(
            api_key="test-api-key",
            cache_dir=tmp_path,
            openai_client=mock_openai
        )

    def test_init_no_api_key_raises_error(self):
        """Test that PerplexityClient raises ValueError if no API key is provided."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="API key must be provided"):
                PerplexityClient(api_key=None)

    def test_init_with_external_openai_client(self):
        """Test that PerplexityClient can be initialized with an external OpenAI client."""
        mock_openai_client = MagicMock()
        client = PerplexityClient(api_key="test-key", openai_client=mock_openai_client)
        assert client.client is mock_openai_client

    def test_classify_pharmacy_invalid_input_type(self, client):
        """Test that classify_pharmacy raises ValueError for invalid input types."""
        with pytest.raises(ValueError, match="pharmacy_data must be a dictionary or PharmacyData object"):
            client.classify_pharmacy(12345)

    def test_classify_pharmacy_missing_name_field(self, client):
        """Test that classify_pharmacy raises ValueError if 'name' is missing."""
        with pytest.raises(ValueError, match="pharmacy_data must contain at least a 'name' field"):
            client.classify_pharmacy({'address': '123 Main St'})

    def test_caching(self, client, mock_openai):
        """Test that repeated calls for the same pharmacy use the cache."""
        client.classify_pharmacy(SAMPLE_PHARMACY)
        client.classify_pharmacy(SAMPLE_PHARMACY)
        mock_openai.chat.completions.create.assert_called_once()

    def test_force_reclassification(self, client, mock_openai):
        """Test that force_reclassification bypasses the cache."""
        client.force_reclassification = True
        client.classify_pharmacy(SAMPLE_PHARMACY)
        client.classify_pharmacy(SAMPLE_PHARMACY)
        assert mock_openai.chat.completions.create.call_count == 2

    def test_cache_read_error(self, tmp_path, mock_openai):
        """Test handling of cache read errors."""
        client = PerplexityClient(api_key="test-key", cache_dir=tmp_path, openai_client=mock_openai)
        with patch('builtins.open', mock_open()) as mock_file:
            mock_file.side_effect = IOError("Failed to read cache")
            client.classify_pharmacy(SAMPLE_PHARMACY)
            mock_openai.chat.completions.create.assert_called_once()

    def test_cache_write_error(self, tmp_path, mock_openai):
        """Test handling of cache write errors."""
        client = PerplexityClient(api_key="test-key", cache_dir=tmp_path, openai_client=mock_openai)
        with patch('builtins.open', mock_open()) as mock_file:
            mock_file.return_value.__enter__.side_effect = IOError("Failed to write cache")
            client.classify_pharmacy(SAMPLE_PHARMACY)
            mock_openai.chat.completions.create.assert_called_once()

    def test_retry_on_rate_limit(self, client, mock_openai):
        """Test that the client retries on rate limit errors."""
        mock_error_response = MagicMock()
        mock_error_response.status_code = 429
        mock_openai.chat.completions.create.side_effect = [
            openai.RateLimitError("Rate limit exceeded", response=mock_error_response, body=None),
            mock_openai.chat.completions.create.return_value
        ]
        client.classify_pharmacy(SAMPLE_PHARMACY)
        assert mock_openai.chat.completions.create.call_count == 2

    def test_rate_limit_exceeded(self, client, mock_openai):
        """Test that RateLimitError is raised when retries are exhausted."""
        client.max_retries = 1
        mock_error_response = MagicMock()
        mock_error_response.status_code = 429
        mock_openai.chat.completions.create.side_effect = openai.RateLimitError(
            "Rate limit exceeded", response=mock_error_response, body=None
        )
        with pytest.raises(RateLimitError):
            client.classify_pharmacy(SAMPLE_PHARMACY)
        assert mock_openai.chat.completions.create.call_count == 2

    def test_api_error_handling(self, client, mock_openai):
        """Test that API errors are handled correctly."""
        mock_openai.chat.completions.create.side_effect = openai.APIError("API error", request=None, body=None)
        with pytest.raises(PerplexityAPIError, match="API error: API error"):
            client.classify_pharmacy(SAMPLE_PHARMACY)

    def test_parse_response_invalid_json(self, client):
        """Test that invalid JSON raises ResponseParseError."""
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = '{"invalid": "json"'
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        with pytest.raises(ResponseParseError, match="Failed to parse API response"):
            client._parse_response(mock_response)

    def test_parse_response_not_a_dict(self, client):
        """Test that a response that is not a dict is handled correctly."""
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = '"just a string"'
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        result = client._parse_response(mock_response)
        assert result == {"response": "just a string"}

    def test_cache_key_generation(self):
        """Test that cache keys are generated consistently."""
        key1 = _generate_cache_key(SAMPLE_PHARMACY, "test-model")
        key2 = _generate_cache_key(SAMPLE_PHARMACY.copy(), "test-model")
        assert key1 == key2

        key3 = _generate_cache_key(SAMPLE_PHARMACY, "different-model")
        assert key1 != key3

        different_pharmacy = SAMPLE_PHARMACY.copy()
        different_pharmacy["name"] = "Different Name"
        key4 = _generate_cache_key(different_pharmacy, "test-model")
        assert key1 != key4
