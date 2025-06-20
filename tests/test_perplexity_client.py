"""
Tests for the Perplexity API client and caching layer.
"""
import os
import time
import json
import pytest
from unittest.mock import patch, MagicMock, ANY
from pathlib import Path
from typing import Dict, Any, Optional

import openai
from openai.types.chat import ChatCompletion, ChatCompletionMessage, ChatCompletionMessageParam
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage

from src.classification.perplexity_client import (
    PerplexityClient,
    PerplexityAPIError,
    RateLimitError,
    classify_pharmacy
)
from src.classification.cache import Cache

# Sample pharmacy data for testing
SAMPLE_PHARMACY = {
    "name": "Main Street Pharmacy",
    "address": "123 Main St, Anytown, USA",
    "phone": "(555) 123-4567"
}

# Fixtures
@pytest.fixture
def mock_openai_client():
    with patch('openai.OpenAI') as mock_client:
        yield mock_client.return_value

@pytest.fixture
def temp_cache_dir(tmp_path):
    return tmp_path / "test_cache"

@pytest.fixture
def mock_env(monkeypatch):
    monkeypatch.setenv('PERPLEXITY_API_KEY', 'test-api-key')
    monkeypatch.setenv('PERPLEXITY_MODEL', 'test-model')
    monkeypatch.setenv('PERPLEXITY_BASE_URL', 'https://api.perplexity.ai')
    monkeypatch.setenv('PERPLEXITY_MAX_RETRIES', '3')
    monkeypatch.setenv('PERPLEXITY_TIMEOUT', '30')
    monkeypatch.setenv('PERPLEXITY_MAX_REQUESTS_PER_MINUTE', '100')

# Test classes
class TestPerplexityClient:
    """Test the Perplexity API client."""
    
    def test_init_defaults(self, mock_env):
        """Test client initialization with default values."""
        client = PerplexityClient()
        assert client.api_key == 'test-api-key'
        assert client.model == 'sonar-medium-online'  # Default model
        assert client.base_url == 'https://api.perplexity.ai'
        assert client.max_retries == 3
        assert client.timeout == 30
        assert client.max_requests_per_minute == 100
    
    def test_classify_pharmacy_success(self, mock_openai_client, mock_env):
        """Test successful pharmacy classification."""
        # Setup mock response
        mock_response = MagicMock(spec=ChatCompletion)
        mock_choice = MagicMock(spec=Choice)
        mock_choice.message = MagicMock(spec=ChatCompletionMessage)
        mock_choice.message.content = json.dumps({
            "is_chain": False,
            "confidence": 0.95,
            "reason": "Independent pharmacy"
        })
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(spec=CompletionUsage)
        mock_response.usage.total_tokens = 100
        
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        # Test
        client = PerplexityClient()
        result = client.classify_pharmacy(SAMPLE_PHARMACY)
        
        # Verify
        assert result == {
            "is_chain": False,
            "confidence": 0.95,
            "reason": "Independent pharmacy"
        }
    
    def test_rate_limit_error(self, mock_openai_client, mock_env):
        """Test rate limit error handling."""
        # Setup mock to raise rate limit error
        from openai import RateLimitError
        error = RateLimitError(
            "Rate limit exceeded",
            response=MagicMock(status_code=429),
            body={"error": {"message": "rate_limit_exceeded"}}
        )
        mock_openai_client.chat.completions.create.side_effect = error
        
        # Test with retries disabled
        client = PerplexityClient(max_retries=0)
        with pytest.raises(RateLimitError):
            client.classify_pharmacy({"name": "Test"})
    
    def test_api_error_handling(self, mock_openai_client, mock_env):
        """Test API error handling."""
        # Setup mock to raise API error
        from openai import APIError
        error = APIError(
            "API error",
            request=MagicMock(),
            body={"error": {"message": "api_error"}}
        )
        mock_openai_client.chat.completions.create.side_effect = error
        
        # Test with retries disabled
        client = PerplexityClient(max_retries=0)
        with pytest.raises(PerplexityAPIError) as exc_info:
            client.classify_pharmacy({"name": "Test"})
        assert exc_info.value.error_type == "api_error"
    
    def test_response_parsing(self, mock_openai_client, mock_env):
        """Test response parsing from API."""
        # Setup mock response with invalid JSON
        mock_response = MagicMock(spec=ChatCompletion)
        mock_choice = MagicMock(spec=Choice)
        mock_choice.message = MagicMock(spec=ChatCompletionMessage)
        mock_choice.message.content = "invalid json"
        mock_response.choices = [mock_choice]
        
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        # Test
        client = PerplexityClient()
        with pytest.raises(PerplexityAPIError) as exc_info:
            client.classify_pharmacy(SAMPLE_PHARMACY)
        # Check that it's a response parsing error
        assert "response_parsing" in str(exc_info.value.error_type).lower()
    
    def test_prompt_generation(self, mock_openai_client, mock_env):
        """Test prompt generation for classification."""
        client = PerplexityClient()
        prompt = client._generate_prompt(SAMPLE_PHARMACY)
        
        assert "Main Street Pharmacy" in prompt
        assert "123 Main St" in prompt
        assert "(555) 123-4567" in prompt
        assert "JSON" in prompt


class TestCachingLayer:
    """Test the caching layer."""
    
    def test_cache_set_get(self, temp_cache_dir):
        """Test setting and getting from cache."""
        cache = Cache("test_cache", cache_dir=str(temp_cache_dir), max_size=10, default_ttl=60)
        
        # Test set and get
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Test get with default
        assert cache.get("nonexistent", "default") == "default"
    
    def test_cache_expiration(self, temp_cache_dir):
        """Test cache expiration."""
        cache = Cache("test_cache", cache_dir=str(temp_cache_dir), default_ttl=0.1)  # 100ms TTL
        
        # Set value with short TTL
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Wait for expiration
        time.sleep(0.2)
        assert cache.get("key1") is None
    
    def test_cache_invalidation(self, temp_cache_dir):
        """Test cache invalidation."""
        cache = Cache("test_cache", cache_dir=str(temp_cache_dir))
        
        # Set and invalidate
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        cache.invalidate("key1")
        assert cache.get("key1") is None


class TestIntegration:
    """Test integration with the classifier."""
    
    def test_classifier_with_caching(self, mock_openai_client, temp_cache_dir, mock_env):
        """Test the classifier with caching enabled."""
        # Setup mock response
        mock_response = MagicMock(spec=ChatCompletion)
        mock_choice = MagicMock(spec=Choice)
        mock_choice.message = MagicMock(spec=ChatCompletionMessage)
        mock_choice.message.content = json.dumps({
            "is_chain": False,
            "confidence": 0.95,
            "reason": "Test reason"
        })
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(spec=CompletionUsage)
        mock_response.usage.total_tokens = 100
        
        # Configure the mock to return our response
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        # Clear any existing cache
        cache = Cache("test_cache", cache_dir=str(temp_cache_dir))
        cache.clear()
        
        # First call - should hit API
        result1 = classify_pharmacy(SAMPLE_PHARMACY, cache_dir=str(temp_cache_dir))
        assert result1["is_chain"] is False
        
        # Verify API was called once
        assert mock_openai_client.chat.completions.create.call_count == 1
        
        # Reset the mock call count
        mock_openai_client.chat.completions.create.reset_mock()
        
        # Second call - should hit cache
        result2 = classify_pharmacy(SAMPLE_PHARMACY, cache_dir=str(temp_cache_dir))
        assert result2 == result1
        
        # Verify API was not called again (should use cache)
        assert mock_openai_client.chat.completions.create.call_count == 0, \
            f"Expected 0 API calls, but got {mock_openai_client.chat.completions.create.call_count}"
        
        # Test with different parameters to ensure cache key is working
        different_pharmacy = SAMPLE_PHARMACY.copy()
        different_pharmacy["name"] = "Different Pharmacy"
        
        # This should hit the API again since it's a different pharmacy
        result3 = classify_pharmacy(different_pharmacy, cache_dir=str(temp_cache_dir))
        assert mock_openai_client.chat.completions.create.call_count == 1, \
            f"Expected 1 API call for different pharmacy, but got {mock_openai_client.chat.completions.create.call_count}"
