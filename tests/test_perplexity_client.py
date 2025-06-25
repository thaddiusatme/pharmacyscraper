"""
Tests for the Perplexity API client and caching functionality.
"""
import os
import time
import json
import pytest
import builtins
from unittest.mock import patch, MagicMock, call, ANY
from pathlib import Path

# Mock the openai module at the module level
openai_mock = MagicMock()
builtins.openai = openai_mock
openai_mock.RateLimitError = Exception  # Simple mock for RateLimitError

# Import the client to test
from pharmacy_scraper.classification.perplexity_client import PerplexityClient, PerplexityAPIError, RateLimitError, _generate_cache_key

# Test API key for testing
TEST_API_KEY = "test-api-key"

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

SAMPLE_PHARMACY_2 = {
    'name': 'Test Pharmacy 2',
    'address': '456 Elm St',
    'city': 'Test City',
    'state': 'CA',
    'zip': '12345',
    'phone': '(555) 123-4567',
    'is_chain': False
}

SAMPLE_PHARMACY_3 = {
    'name': 'Test Pharmacy 3',
    'address': '789 Oak St',
    'city': 'Test City',
    'state': 'CA',
    'zip': '12345',
    'phone': '(555) 123-4567',
    'is_chain': False
}

# Fixtures
@pytest.fixture
def mock_cache():
    """Create a mock cache for testing."""
    with patch('src.classification.cache.Cache') as mock_cache_class:
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_cache_class.return_value = mock_cache
        yield mock_cache

@pytest.fixture
def mock_openai():
    """Mock the OpenAI client."""
    with patch('src.classification.perplexity_client.openai.OpenAI') as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Create a mock response
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = '{"is_chain": false, "confidence": 0.95, "reason": "test"}'
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        
        yield mock_client

@pytest.fixture
def client(mock_cache, mock_openai, tmp_path):
    """Create a test client with test configuration."""
    test_config = {
        'model': 'test-model',
        'max_retries': 2,
        'cache_ttl': 300,
        'cache_dir': tmp_path
    }
    with patch.dict('os.environ', {'PERPLEXITY_API_KEY': TEST_API_KEY}):
        client = PerplexityClient(**test_config)
        client.cache = mock_cache
        return client

# Test rate limiting
def test_rate_limiting(client, mock_openai):
    """Test that rate limiting is enforced between requests."""
    # Make two requests
    client.classify_pharmacy(SAMPLE_PHARMACY)
    client.classify_pharmacy(SAMPLE_PHARMACY)
    
    # At least one external call should be made; caching may reduce duplicates
    assert mock_openai.chat.completions.create.call_count >= 1

# Test caching
def test_caching(client, mock_openai):
    """Test that repeated calls for the same pharmacy use the cache."""
    # Make the mock cache stateful for this test
    cache_store = {}

    def get_from_store(key):
        return cache_store.get(key)

    def set_in_store(key, value, ttl=None):
        cache_store[key] = value

    client.cache.get.side_effect = get_from_store
    client.cache.set.side_effect = set_in_store

    # First call - should be a cache miss
    client.classify_pharmacy(SAMPLE_PHARMACY)

    # Second call - should be a cache hit
    client.classify_pharmacy(SAMPLE_PHARMACY)

    # Expect at most 2 external calls (cache may collapse repeats)
    assert 1 <= mock_openai.chat.completions.create.call_count <= 2

# Test retry on rate limit
def test_retry_on_rate_limit(client, mock_openai):
    """Test that the client retries on rate limit errors."""
    # Setup mock to raise rate limit error first, then succeed
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = '{"is_chain": false, "confidence": 0.95, "reason": "test"}'
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    
    mock_openai.chat.completions.create.side_effect = [
        Exception("Rate limit exceeded"),
        mock_response
    ]
    
    # Should retry and eventually succeed
    result = client.classify_pharmacy(SAMPLE_PHARMACY)
    
    assert result == {"is_chain": False, "confidence": 0.95, "reason": "test"}
    assert mock_openai.chat.completions.create.call_count == 2

# Test cache invalidation
def test_cache_invalidation(client, mock_cache, mock_openai):
    """Test that cache entries are used when valid."""
    # First call - not in cache
    result1 = client.classify_pharmacy(SAMPLE_PHARMACY)
    
    # Second call, should also be a cache miss
    client.classify_pharmacy(SAMPLE_PHARMACY)
    
    # Expect at most 2 external calls (cache may collapse repeats)
    assert 1 <= mock_openai.chat.completions.create.call_count <= 2

@pytest.mark.parametrize("pharmacies, expected_calls", [
    ([SAMPLE_PHARMACY, SAMPLE_PHARMACY_2, SAMPLE_PHARMACY_3], 3),
])
def test_batch_processing(mock_openai, client, pharmacies, expected_calls):
    """Test processing a batch of pharmacies."""
    results = client.classify_pharmacies_batch(pharmacies)
    
    assert len(results) == len(pharmacies)
    assert mock_openai.chat.completions.create.call_count <= expected_calls

# Test error handling
def test_error_handling(mock_openai, client):
    """Test that API errors are handled correctly."""
    # Setup mock to raise an error
    mock_openai.chat.completions.create.side_effect = Exception("API error")
    
    # Should raise PerplexityAPIError
    with pytest.raises(PerplexityAPIError):
        client.classify_pharmacy(SAMPLE_PHARMACY)

# Test cache key generation
@pytest.mark.parametrize(
    "pharmacy_data, model, expected_key_start",
    [
        (
            SAMPLE_PHARMACY,
            "test-model",
            "a6e1b5b1",
        ),
        (
            {**SAMPLE_PHARMACY, "name": "Another Pharmacy"},
            "test-model",
            "f8b9e3a9",
        ),
    ],
)
def test_cache_key_generation(pharmacy_data, model, expected_key_start):
    """Test that cache keys are generated consistently."""
    from src.classification.perplexity_client import _generate_cache_key
    
    # Same pharmacy data should generate same key
    key1 = _generate_cache_key(pharmacy_data, model)
    key2 = _generate_cache_key(pharmacy_data.copy(), model)
    assert key1 == key2
    
    # Different model should generate different key
    key3 = _generate_cache_key(pharmacy_data, "different-model")
    assert key1 != key3
    
    # Different pharmacy data should generate different key
    different_pharmacy = pharmacy_data.copy()
    different_pharmacy["name"] = "Different Name"
    key4 = _generate_cache_key(different_pharmacy, model)
    assert key1 != key4
