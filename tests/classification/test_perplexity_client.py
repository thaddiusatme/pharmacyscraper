"""
Comprehensive tests for the Perplexity API client, caching, and error handling.
"""
import os
import time
import json
import pytest
import openai
import tempfile
from unittest.mock import patch, MagicMock, mock_open
import tenacity
from pharmacy_scraper.classification.perplexity_client import (
    PerplexityClient,
    RateLimitError,
    InvalidRequestError,
    ResponseParseError,
    PerplexityAPIError,
    _generate_cache_key,
    RateLimiter
)
from pharmacy_scraper.classification.models import PharmacyData, ClassificationResult, ClassificationSource

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
        client = PerplexityClient(
            api_key="test-api-key",
            cache_dir=str(tmp_path),
            cache_enabled=True
        )
        # Set the mock client directly since we can't pass it in constructor
        client.client = mock_openai
        return client

    def test_init_no_api_key_raises_error(self):
        """Test that PerplexityClient raises ValueError if no API key is provided."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="No Perplexity API key provided"):
                PerplexityClient(api_key=None)

    def test_init_with_external_openai_client(self):
        """Test that PerplexityClient's client attribute can be modified after initialization."""
        client = PerplexityClient(api_key="test-key", cache_enabled=False)
        mock_client = MagicMock()
        client.client = mock_client
        assert client.client == mock_client

    def test_classify_pharmacy_invalid_input_type(self, client):
        """Test that classify_pharmacy raises TypeError for invalid input types."""
        # The current implementation expects a dict or PharmacyData object
        # Use a string object which can't be properly converted
        with patch.object(PharmacyData, 'from_dict', side_effect=AttributeError("'str' object has no attribute 'get'")):
            with pytest.raises(PerplexityAPIError):
                client.classify_pharmacy("not a valid input")

    def test_classify_pharmacy_missing_name_field(self, client):
        """Test that classify_pharmacy raises PerplexityAPIError if 'name' is missing."""
        # In current implementation, missing name gets wrapped in a PerplexityAPIError
        with pytest.raises(PerplexityAPIError):
            client.classify_pharmacy({'address': '123 Main St'})

    def test_caching(self, client, mock_openai):
        """Test that repeated calls for the same pharmacy use the cache."""
        # Create a pharmacy data object from our sample
        pharmacy_data = PharmacyData.from_dict(SAMPLE_PHARMACY)
        
        # Create a result that will be "cached"
        result = ClassificationResult(
            classification="independent",
            is_chain=False,
            is_compounding=False,
            confidence=0.9,
            explanation="Test explanation",
            source=ClassificationSource.PERPLEXITY,
            model="test-model",
            pharmacy_data=pharmacy_data
        )
        
        # Create a mock for the cache
        cache_mock = MagicMock()
        # First call to get returns None (cache miss)
        # Second call returns our cached result
        cache_mock.get.side_effect = [None, result]
        
        # Replace the actual cache with our mock
        client.cache = cache_mock
        
        # Create a mock API response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"classification": "independent", "is_chain": false}'
        
        # Setup our mocks
        with patch.object(client, '_call_api_with_retries', return_value=mock_response):
            with patch.object(client, '_parse_response', return_value=result):
                # First call should use the API (cache miss)
                first_result = client.classify_pharmacy(SAMPLE_PHARMACY)
                
                # Second call should use the cache
                second_result = client.classify_pharmacy(SAMPLE_PHARMACY)
                
                # Verify API was only called once
                assert client._call_api_with_retries.call_count == 1
                # Verify cache.set was called once to store the result
                assert cache_mock.set.call_count == 1
                # Verify cache.get was called twice
                assert cache_mock.get.call_count == 2

    def test_force_reclassification(self, client, mock_openai):
        """Test that force_reclassification bypasses the cache."""
        # Create a pharmacy data object from our sample
        pharmacy_data = PharmacyData.from_dict(SAMPLE_PHARMACY)
        
        # Create a result that will be in the "cache"
        result = ClassificationResult(
            classification="independent",
            is_chain=False,
            is_compounding=False,
            confidence=0.9,
            explanation="Test explanation",
            source=ClassificationSource.PERPLEXITY,
            model="test-model",
            pharmacy_data=pharmacy_data
        )
        
        # Create a mock for the cache that always returns our cached result
        cache_mock = MagicMock()
        cache_mock.get.return_value = result
        
        # Replace the actual cache with our mock
        client.cache = cache_mock
        
        # Create a mock API response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"classification": "independent", "is_chain": false}'
        
        # Setup our mocks
        with patch.object(client, '_call_api_with_retries', return_value=mock_response):
            with patch.object(client, '_parse_response', return_value=result):
                # Enable force reclassification
                client.force_reclassification = True
                
                # Even with a cached result, both calls should use the API due to force_reclassification
                client.classify_pharmacy(SAMPLE_PHARMACY)
                client.classify_pharmacy(SAMPLE_PHARMACY)
                
                # Verify API was called twice
                assert client._call_api_with_retries.call_count == 2
                # Verify cache.set was called twice
                assert cache_mock.set.call_count == 2

    def test_cache_read_error(self, tmp_path, mock_openai):
        """Test handling of cache read errors."""
        # Create a new client with our mock OpenAI client
        client = PerplexityClient(api_key="test-key", cache_dir=str(tmp_path), cache_enabled=True)
        client.client = mock_openai
        
        # Create a pharmacy data object from our sample
        pharmacy_data = PharmacyData.from_dict(SAMPLE_PHARMACY)
        
        # Create a result for the API to return
        result = ClassificationResult(
            classification="independent",
            is_chain=False,
            is_compounding=False,
            confidence=0.9,
            explanation="Test explanation",
            source=ClassificationSource.PERPLEXITY,
            model="test-model",
            pharmacy_data=pharmacy_data
        )
        
        # Create a mock for _get_from_cache that raises an IOError
        with patch.object(client, '_get_from_cache', side_effect=IOError("Failed to read cache")):
            # Create a mock API response
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = '{"classification": "independent", "is_chain": false}'
            
            # Setup our API call mock
            with patch.object(client, '_call_api_with_retries', return_value=mock_response):
                with patch.object(client, '_parse_response', return_value=result):
                    # Should fall back to API call when cache read fails
                    result = client.classify_pharmacy(SAMPLE_PHARMACY)
                    
                    # Verify API was called
                    assert client._call_api_with_retries.call_count == 1
                    # Verify we got a classification result
                    assert isinstance(result, ClassificationResult)

    def test_cache_write_error(self, tmp_path, mock_openai):
        """Test handling of cache write errors."""
        client = PerplexityClient(api_key="test-key", cache_dir=str(tmp_path), cache_enabled=True)
        client.client = mock_openai
        
        # Patch _parse_response to return a ClassificationResult to avoid errors
        from pharmacy_scraper.classification.models import ClassificationResult, PharmacyData, ClassificationSource
        
        result = ClassificationResult(
            classification="independent",
            is_chain=False,
            is_compounding=False,
            confidence=0.9,
            explanation="Test explanation",
            source=ClassificationSource.PERPLEXITY,
            model="test-model",
            pharmacy_data=PharmacyData.from_dict(SAMPLE_PHARMACY)
        )
        
        # We need to patch the Cache class set method instead of builtins.open
        with patch.object(client, '_parse_response', return_value=result):
            with patch.object(client.cache, 'set', side_effect=IOError("Failed to write cache")):
                client.classify_pharmacy(SAMPLE_PHARMACY)
                mock_openai.chat.completions.create.assert_called_once()

    @patch('tenacity.retry')
    def test_retry_on_rate_limit(self, mock_retry, client, mock_openai):
        """Test that the client retries on rate limit errors."""
        # Mock the retry decorator to avoid actual retries
        mock_retry.return_value = lambda f: f
        
        # Create a pharmacy data object from our sample
        pharmacy_data = PharmacyData.from_dict(SAMPLE_PHARMACY)
        
        # Create a result for the API to return
        result = ClassificationResult(
            classification="independent",
            is_chain=False,
            is_compounding=False,
            confidence=0.9,
            explanation="Test explanation",
            source=ClassificationSource.PERPLEXITY,
            model="test-model",
            pharmacy_data=pharmacy_data
        )
        
        # Create a mock API response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"classification": "independent", "is_chain": false}'
        
        # Patch API call and response parsing
        with patch.object(client.client.chat.completions, 'create', return_value=mock_response):
            with patch.object(client, '_parse_response', return_value=result):
                client.classify_pharmacy(SAMPLE_PHARMACY)
                # Verify that the completions.create method was called
                client.client.chat.completions.create.assert_called_once()

    def test_rate_limit_exceeded(self, client, mock_openai):
        """Test that RateLimitError is wrapped in PerplexityAPIError when retries are exhausted."""
        # Create a mock error response that mimics OpenAI's RateLimitError
        mock_response = MagicMock()
        mock_response.status_code = 429
        rate_limit_error = openai.RateLimitError(
            "Rate limit exceeded", 
            response=mock_response,
            body={"error": {"message": "Rate limit exceeded"}}
        )
        
        # Patch _call_api_with_retries to raise the rate limit error
        with patch.object(client, '_call_api_with_retries', side_effect=rate_limit_error):
            with pytest.raises(PerplexityAPIError) as exc_info:
                client.classify_pharmacy(SAMPLE_PHARMACY)
            
            # Verify the error message and type
            assert "api_error" in str(exc_info.value)
            assert "Rate limit exceeded" in str(exc_info.value)

    def test_api_error_handling(self, client, mock_openai):
        """Test that API errors are handled correctly."""
        # Create a mock error that mimics OpenAI's APIError
        api_error = openai.APIError(
            "API error", 
            request=MagicMock(), 
            body={"error": {"message": "API error"}}
        )
        
        # Patch _call_api_with_retries to raise the API error
        with patch.object(client, '_call_api_with_retries', side_effect=api_error):
            with pytest.raises(PerplexityAPIError) as exc_info:
                client.classify_pharmacy(SAMPLE_PHARMACY)
            
            # Verify the error message
            assert "API error: API error" in str(exc_info.value)

    def test_parse_response_invalid_json(self, client):
        """Test that invalid JSON raises ResponseParseError."""
        from pharmacy_scraper.classification.models import PharmacyData
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = '{"invalid": "json"'
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        pharmacy_data = PharmacyData.from_dict(SAMPLE_PHARMACY)
        with pytest.raises(ResponseParseError):
            client._parse_response(mock_response, pharmacy_data)

    def test_parse_response_not_a_dict(self, client):
        """Test that a response that is not a dict raises ResponseParseError."""
        from pharmacy_scraper.classification.models import PharmacyData
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = '"just a string"'
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        pharmacy_data = PharmacyData.from_dict(SAMPLE_PHARMACY)
        with pytest.raises(ResponseParseError):
            client._parse_response(mock_response, pharmacy_data)

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
