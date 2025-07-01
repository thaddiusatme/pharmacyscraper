"""
Comprehensive tests for the PerplexityClient class.

This test suite focuses on areas not covered by existing tests and aims to improve
overall test coverage for the Perplexity client.
"""
import os
import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, ANY
import pytest
from openai import RateLimitError as OpenAIRateLimitError

# Mock the required OpenAI types if they're not available
try:
    from openai.types import APIResponse
except ImportError:
    class APIResponse:
        pass

try:
    from openai.types.chat import ChatCompletion, ChatCompletionMessage, ChatCompletionChunk
except ImportError:
    class ChatCompletion:
        pass
    class ChatCompletionMessage:
        pass
    class ChatCompletionChunk:
        pass

try:
    from openai.types.chat.chat_completion import Choice
except ImportError:
    class Choice:
        pass

try:
    from openai.types.completion_usage import CompletionUsage
except ImportError:
    class CompletionUsage:
        pass

from src.pharmacy_scraper.classification.perplexity_client import (
    PerplexityClient,
    PerplexityAPIError,
    RateLimiter,
    RateLimitError,
    _generate_cache_key
)

# Sample test data
SAMPLE_PHARMACY = {
    'name': 'Test Pharmacy',
    'address': '123 Main St',
    'city': 'Test City',
    'state': 'CA',
    'zip': '12345'
}

SAMPLE_RESPONSE = {
    "classification": "independent",
    "is_compounding": False,
    "confidence": 0.9,
    "explanation": "This is a test explanation."
}


class TestPerplexityClientInitialization:
    """Tests for PerplexityClient initialization and configuration."""
    
    def test_init_with_api_key(self, tmp_path):
        """Test initialization with explicit API key."""
        client = PerplexityClient(
            api_key="test-key",
            model_name="test-model",
            rate_limit=10,
            cache_dir=str(tmp_path)
        )
        
        assert client.model_name == "test-model"
        assert client.rate_limiter.min_interval == 6.0  # 60s / 10 requests
        assert client.cache.cache_dir == str(tmp_path)
    
    def test_init_without_api_key_raises(self):
        """Test that initialization fails without API key."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="Perplexity API key not found"):
                PerplexityClient()
    
    def test_init_with_environment_variable(self):
        """Test initialization with API key from environment variable."""
        with patch.dict('os.environ', {'PERPLEXITY_API_KEY': 'env-key'}):
            client = PerplexityClient()
            assert client.api_key == 'env-key'


class TestCacheFunctionality:
    """Tests for cache-related functionality."""
    
    def test_cache_miss_then_hit(self, tmp_path, mock_openai):
        """Test that cache is populated on miss and used on subsequent calls."""
        # Setup
        cache_dir = tmp_path / "test_cache"
        cache_dir.mkdir()
        
        # Mock API response
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = json.dumps(SAMPLE_RESPONSE)
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        
        # First call - should call API and cache result
        client = PerplexityClient(
            api_key="test-key",
            cache_dir=str(cache_dir),
            openai_client=mock_client,
            model_name="test-model" # ensure model is consistent
        )
        
        # First call - should hit API
        result1 = client.classify_pharmacy(SAMPLE_PHARMACY)
        assert mock_client.chat.completions.create.call_count == 1
        
        # Second call - should use cache
        result2 = client.classify_pharmacy(SAMPLE_PHARMACY)
        assert mock_client.chat.completions.create.call_count == 1  # Still 1
        assert result1 == result2
        
        # Verify cache file was created
        cache_key = _generate_cache_key(SAMPLE_PHARMACY, client.model_name)
        cache_file = Path(cache_dir) / f"{cache_key}.json"
        assert cache_file.exists()
        
        # Verify cache contents
        with open(cache_file, 'r') as f:
            cached_data = json.load(f)
        assert cached_data == SAMPLE_RESPONSE
    
    def test_force_reclassification(self, tmp_path, mock_openai):
        """Test that force_reclassification ignores cache."""
        # Setup cache with existing entry
        cache_dir = tmp_path / "test_cache"
        cache_dir.mkdir()
        client_for_key = PerplexityClient(api_key="test-key", model_name="test-model")
        cache_key = _generate_cache_key(SAMPLE_PHARMACY, client_for_key.model_name)
        cache_file = cache_dir / f"{cache_key}.json"
        
        with open(cache_file, 'w') as f:
            json.dump({"cached": True}, f)
        
        # Mock API response
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = json.dumps(SAMPLE_RESPONSE)
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        
        # Create client with force_reclassification=True
        client = PerplexityClient(
            api_key="test-key",
            cache_dir=str(cache_dir),
            force_reclassification=True,
            openai_client=mock_client,
            model_name="test-model"
        )
        
        # Should ignore cache and call API
        result = client.classify_pharmacy(SAMPLE_PHARMACY)
        assert mock_client.chat.completions.create.called
        assert result == SAMPLE_RESPONSE


class TestErrorHandling:
    """Tests for error handling in the Perplexity client."""
    
    def test_api_error_handling(self, mock_openai, tmp_path):
        """Test handling of API errors with retries."""
        # Setup mock to fail with rate limit error, then succeed
        mock_client = MagicMock()
        
        # Create a proper RateLimitError with required parameters
        rate_limit_error = OpenAIRateLimitError(
            message="Rate limit exceeded",
            response=MagicMock(status_code=429),
            body={"error": {"message": "Rate limit exceeded"}}
        )
        
        # Create a mock success response
        success_response = MagicMock()
        success_choice = MagicMock()
        success_message = MagicMock()
        success_message.content = json.dumps(SAMPLE_RESPONSE)
        success_choice.message = success_message
        success_response.choices = [success_choice]
        
        # Set up side effect
        mock_client.chat.completions.create.side_effect = [
            rate_limit_error,
            success_response
        ]
        
        # Use a fresh cache directory for this test
        cache_dir = tmp_path / "test_cache"
        cache_dir.mkdir()
        
        client = PerplexityClient(
            api_key="test-key",
            openai_client=mock_client,
            max_retries=3,
            cache_dir=str(cache_dir),
            force_reclassification=True  # Ensure we don't use cached results
        )
        
        # Clear any cached results before the test
        if client.cache.cache_dir:
            client.cache.clear()
        
        # Should retry and eventually succeed
        result = client.classify_pharmacy(SAMPLE_PHARMACY)
        assert result == SAMPLE_RESPONSE
        assert mock_client.chat.completions.create.call_count == 2
    
    def test_max_retries_exceeded(self, mock_openai, tmp_path):
        """Test that max retries are respected."""
        # Setup mock to always fail
        mock_client = MagicMock()
        
        # Create a proper RateLimitError with required parameters
        rate_limit_error = OpenAIRateLimitError(
            message="Rate limit exceeded",
            response=MagicMock(status_code=429),
            body={"error": {"message": "Rate limit exceeded"}}
        )
        
        mock_client.chat.completions.create.side_effect = rate_limit_error
        
        # Use a fresh cache directory for this test
        cache_dir = tmp_path / "test_cache"
        cache_dir.mkdir()
        
        client = PerplexityClient(
            api_key="test-key",
            openai_client=mock_client,
            max_retries=2,
            cache_dir=str(cache_dir),
            force_reclassification=True  # Ensure we don't use cached results
        )
        
        # Clear any cached results before the test
        if client.cache.cache_dir:
            client.cache.clear()

        # Should raise after max retries
        with pytest.raises(RateLimitError, match=f"Rate limit exceeded after {client.max_retries} retries"):
            client.classify_pharmacy(SAMPLE_PHARMACY)
            
        # Verify the API was called the expected number of times (max_retries + 1)
        assert mock_client.chat.completions.create.call_count == client.max_retries + 1


class TestRateLimiter:
    """Tests for the RateLimiter class."""
    
    def test_rate_limiter_with_multiple_waits(self):
        """Test that rate limiter enforces delays between requests."""
        limiter = RateLimiter(60)  # 1 request per second (min_interval = 1.0)
        
        # First call should not wait
        start = time.time()
        limiter.wait()
        first_call_time = time.time() - start
        assert first_call_time < 0.1  # Should be almost instant
        
        # Second call should wait about 1 second
        start = time.time()
        limiter.wait()
        second_call_time = time.time() - start
        assert 0.9 <= second_call_time <= 1.1  # Allow some tolerance

    def test_rate_limiter_no_wait_for_zero_rate_limit(self):
        """Test that a rate limit of 0 or less results in no waiting."""
        limiter = RateLimiter(0)
        start = time.time()
        limiter.wait()
        limiter.wait()
        duration = time.time() - start
        assert duration < 0.01

    def test_rate_limiter_initial_call_is_immediate(self):
        """Test that the first call to wait() is always immediate."""
        limiter = RateLimiter(1) # 1 request per minute
        start = time.time()
        limiter.wait()
        duration = time.time() - start
        assert duration < 0.01

    @patch('time.sleep', return_value=None)
    def test_rate_limiter_sleep_calculation(self, mock_sleep):
        """Test that time.sleep is called with the correct delay."""
        rate_limit = 120  # 2 requests per second
        limiter = RateLimiter(rate_limit) # min_interval = 0.5
        
        # First call, no sleep
        limiter.wait()
        mock_sleep.assert_not_called()

        # Second call immediately after, should sleep for ~0.5s
        limiter.wait()
        mock_sleep.assert_called_once()
        # Check that sleep was called with a value close to 0.5
        assert 0.49 < mock_sleep.call_args[0][0] < 0.51


# Fixtures
@pytest.fixture
def mock_openai():
    """Fixture to mock the OpenAI client."""
    with patch('pharmacy_scraper.classification.perplexity_client.OpenAI') as mock_openai_class:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = json.dumps(SAMPLE_RESPONSE)
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        yield mock_client
