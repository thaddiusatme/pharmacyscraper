"""
Tests for edge cases and additional coverage in the Perplexity client.
"""
import time
import json
import pytest
from unittest.mock import patch, MagicMock, ANY
from pharmacy_scraper.classification.perplexity_client import (
    PerplexityClient, 
    RateLimiter,
    PerplexityAPIError,
    RateLimitError,
    InvalidRequestError,
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

class TestRateLimiter:
    """Tests for the RateLimiter class."""
    
    def test_rate_limiter_no_limit(self):
        """Test that no delay occurs when rate limit is 0."""
        limiter = RateLimiter(0)
        start_time = time.time()
        limiter.wait()
        end_time = time.time()
        assert end_time - start_time < 0.1  # Should be almost instant
    
    def test_rate_limiter_with_limit(self):
        """Test that rate limiting enforces minimum delay between requests."""
        limiter = RateLimiter(60)  # 1 request per second
        start_time = time.time()
        limiter.wait()  # First call should not wait
        limiter.wait()  # Second call should wait ~1 second
        end_time = time.time()
        assert end_time - start_time >= 1.0

class TestPerplexityClientEdgeCases:
    """Tests for edge cases in the PerplexityClient class."""
    
    @patch('pharmacy_scraper.classification.perplexity_client.OpenAI')
    def test_call_api_success(self, mock_openai):
        """Test successful API call with valid response."""
        # Setup mock response
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        
        # Mock the response structure that _parse_response expects
        mock_message.content = '{"is_pharmacy": true, "is_compound_pharmacy": false, "confidence": 0.95}'
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        
        # Configure the mock client to return our mock response
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Create client with mocked OpenAI
        client = PerplexityClient(api_key="test-key", openai_client=mock_client)
        
        # Call the method under test
        response = client._call_api("test prompt")
        
        # Verify the response is parsed correctly
        assert isinstance(response, dict)
        assert response.get('is_pharmacy') is True
        assert response.get('confidence') == 0.95
        mock_client.chat.completions.create.assert_called_once()
    
    @patch('pharmacy_scraper.classification.perplexity_client.OpenAI')
    def test_call_api_retry_success(self, mock_openai):
        """Test that API retries on rate limit error."""
        # Setup mock to fail once then succeed
        mock_client = MagicMock()
        
        # Create a mock that will raise an exception on first call and return a response on second
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = '{"is_pharmacy": true, "confidence": 0.9}'
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        
        # Set up side effect to raise an exception first, then return the mock response
        mock_client.chat.completions.create.side_effect = [
            Exception("Rate limit"),  # First call fails
            mock_response  # Second call succeeds
        ]
        
        mock_openai.return_value = mock_client
        
        # Create a client with max_retries=1 (will retry once)
        client = PerplexityClient(api_key="test-key", max_retries=1, openai_client=mock_client)
        
        # This should not raise an exception
        response = client._call_api("test prompt")
        
        # Should have been called twice (initial + 1 retry)
        assert mock_client.chat.completions.create.call_count == 2
        assert response == {"is_pharmacy": True, "confidence": 0.9}
    
    @patch('pharmacy_scraper.classification.perplexity_client.OpenAI')
    def test_call_api_rate_limit_exceeded(self, mock_openai):
        """Test that RateLimitError is raised when retries are exhausted."""
        # Setup mock to always raise a rate limit error
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("Rate limit")
        mock_openai.return_value = mock_client
        
        # Create a client with max_retries=2 and the mocked client
        client = PerplexityClient(api_key="test-key", max_retries=2, openai_client=mock_client)
        
        # Should raise RateLimitError after all retries are exhausted
        with pytest.raises(RateLimitError):
            client._call_api("test prompt")
        
        assert mock_client.chat.completions.create.call_count == 3  # Initial + 2 retries
    
    def test_generate_prompt(self):
        """Test prompt generation from pharmacy data."""
        client = PerplexityClient(api_key="test-key")
        prompt = client._generate_prompt(SAMPLE_PHARMACY)
        
        # The prompt should be a string and contain the pharmacy name
        assert isinstance(prompt, str)
        assert SAMPLE_PHARMACY['name'] in prompt
        
        # The prompt should follow the few-shot examples format
        assert "```json" in prompt
        assert "classification" in prompt
        assert "is_compounding" in prompt
        assert "confidence" in prompt
    
    @patch('pharmacy_scraper.classification.perplexity_client.PerplexityClient._call_api')
    def test_parse_response_error(self, mock_call_api, tmp_path):
        """Test handling of invalid API response format."""
        # Setup mock to return None to simulate a parsing error
        mock_call_api.return_value = None
        
        # Create a client with a fresh cache directory and caching disabled
        cache_dir = tmp_path / "test_cache"
        client = PerplexityClient(
            api_key="test-key",
            cache_dir=cache_dir,
            force_reclassification=True  # Ensure we don't use any cached results
        )
        
        # classify_pharmacy should return None when _call_api returns None
        result = client.classify_pharmacy(SAMPLE_PHARMACY)
        
        # Verify the mock was called
        mock_call_api.assert_called_once()
        
        # The result should be None when API call fails
        assert result is None
    
    @patch('pharmacy_scraper.classification.perplexity_client.PerplexityClient._call_api')
    def test_classify_pharmacy_invalid_input(self, mock_call_api):
        """Test handling of invalid input data."""
        # Create a client with caching disabled to ensure our mock is called
        client = PerplexityClient(api_key="test-key", cache_dir=None)
        
        # Mock the _call_api to return a valid response when called
        mock_call_api.return_value = {'is_pharmacy': False, 'confidence': 0.0}
        
        # Test with empty dict (should work)
        result = client.classify_pharmacy({})
        assert isinstance(result, dict)
        assert 'is_pharmacy' in result
        
        # Test with None - should raise an AttributeError
        with pytest.raises(AttributeError):
            client.classify_pharmacy(None)
        
        # Verify the API was called only once (for the empty dict case)
        # The None case should fail before making an API call
        assert mock_call_api.call_count == 1

class TestErrorHandling:
    """Tests for error handling in the Perplexity client."""
    
    @patch('openai.OpenAI')
    def test_invalid_api_key(self, mock_openai):
        """Test handling of invalid API key."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("Invalid API key")
        mock_openai.return_value = mock_client
        
        client = PerplexityClient(api_key="invalid-key")
        
        with pytest.raises(PerplexityAPIError):
            client._call_api("test prompt")
    
    @patch('openai.OpenAI')
    def test_network_error(self, mock_openai):
        """Test handling of network errors."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = ConnectionError("Network error")
        mock_openai.return_value = mock_client
        
        client = PerplexityClient(api_key="test-key", max_retries=1)
        
        with pytest.raises(PerplexityAPIError):
            client._call_api("test prompt")
