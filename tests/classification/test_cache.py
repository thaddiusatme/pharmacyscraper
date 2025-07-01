"""Tests for the cache functionality in PerplexityClient."""
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, ANY
import pytest

from pharmacy_scraper.classification.perplexity_client import (
    PerplexityClient,
    _generate_cache_key,
)
from pharmacy_scraper.classification.models import (
    ClassificationResult,
    ClassificationSource,
    ClassificationMethod,
    PharmacyData
)
from pharmacy_scraper.utils.cache import Cache

# Sample data for testing
SAMPLE_PHARMACY = {
    "name": "Test Pharmacy",
    "address": "123 Test St, Test City, TS 12345"
}

SAMPLE_RESPONSE_DICT = {
    "classification": "independent",
    "reasoning": "Test pharmacy is independently owned",
    "confidence": 0.95
}

# Create a dictionary representation of the expected response
SAMPLE_RESPONSE_DICT = {
    "classification": "independent",
    "is_chain": False,
    "is_compounding": True,
    "confidence": 0.95,
    "explanation": "Test pharmacy is independently owned",
    "source": ClassificationSource.PERPLEXITY,
    "cached": False  # This is a property computed from source
}

# Create a ClassificationResult instance with the correct enum values
SAMPLE_RESPONSE = ClassificationResult(
    classification="independent",
    is_chain=False,
    is_compounding=True,
    confidence=0.95,
    explanation="Test pharmacy is independently owned",
    source=ClassificationSource.PERPLEXITY
    # method parameter is no longer used
)

# Expected cached data structure
EXPECTED_CACHED_DATA = {
    "value": SAMPLE_RESPONSE_DICT,
    "expires_at": None  # TTL is disabled in tests
}


@pytest.fixture
def client_fixture(tmp_path):
    """Provides a PerplexityClient with a patched API call and a temporary cache directory."""
    # Create cache directory
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(exist_ok=True)
    
    with patch.object(
        PerplexityClient, "_make_api_call", return_value=SAMPLE_RESPONSE_DICT
    ) as mock_make_api_call:
        client = PerplexityClient(
            api_key="fake_key",
            cache_dir=str(cache_dir),
            cache_ttl_seconds=None  # Disable TTL for tests
        )
        yield client, mock_make_api_call, cache_dir


@pytest.mark.usefixtures("client_fixture")
class TestCache:
    def test_cache_miss_and_write(self, client_fixture, tmp_path):
        """Test that a cache miss triggers an API call and writes the result to the cache."""
        client, mock_make_api_call, cache_dir = client_fixture

        # Create a simplified response dictionary that matches what the API would return
        api_response = {
            'is_chain': False,
            'is_compounding': True,
            'confidence': 0.95,
            'reason': 'Test pharmacy is independently owned',
            'source': 'perplexity',
            'method': 'llm'
        }
        
        # Configure the mock to return our sample response
        mock_make_api_call.return_value = api_response

        # First call - should miss cache and call API
        result = client.classify_pharmacy(SAMPLE_PHARMACY)

        # Verify the API was called and we got the expected result
        mock_make_api_call.assert_called_once()
        
        # Convert the result to a dict for comparison
        result_dict = dict(result._asdict()) if hasattr(result, '_asdict') else dict(result)
        expected_dict = dict(SAMPLE_RESPONSE_DICT)
        
        # Compare the dictionaries
        assert result_dict == expected_dict, f"Expected {expected_dict}, got {result_dict}"

        # Generate the expected cache key
        cache_key = _generate_cache_key(SAMPLE_PHARMACY, client.model_name)
        cache_file = cache_dir / f"{cache_key}.json"
        
        # Check if cache file exists
        assert cache_file.exists(), f"Cache file {cache_file} does not exist"

        # Read and verify cached data
        with open(cache_file, "r") as f:
            cached_data = json.load(f)

        # Check that the response data is cached correctly
        assert 'value' in cached_data, "Cached data missing 'value' key"
        assert 'expires_at' in cached_data, "Cached data missing 'expires_at' key"
        
        # Check the structure of the cached value
        cached_value = cached_data['value']
        assert isinstance(cached_value, dict), "Cached value should be a dictionary"
        
        # Verify all expected fields are present
        expected_keys = {
            'is_chain': bool,
            'is_compounding': bool,
            'confidence': (int, float),
            'reason': str,
            'source': str,
            'method': str
        }
        
        for key, expected_type in expected_keys.items():
            assert key in cached_value, f"Cached value missing '{key}'"
            assert isinstance(cached_value[key], expected_type), \
                f"Expected {key} to be {expected_type}, got {type(cached_value[key])}"
        
        # Verify the values match our expected response
        for key, expected_value in SAMPLE_RESPONSE_DICT.items():
            assert cached_value[key] == expected_value, \
                f"Mismatch for {key}: expected {expected_value}, got {cached_value[key]}"

    def test_cache_hit(self, client_fixture, caplog):
        """Test that a subsequent call for the same data hits the cache."""
        client, mock_make_api_call, _ = client_fixture
        
        # Create a simplified response dictionary that matches what the API would return
        api_response = {
            'is_chain': False,
            'is_compounding': True,
            'confidence': 0.95,
            'reason': 'Test pharmacy is independently owned',
            'source': 'perplexity',
            'method': 'llm'
        }
        
        # Configure the mock to return our sample response
        mock_make_api_call.return_value = api_response
        
        # Enable debug logging
        import logging
        logger = logging.getLogger('pharmacy_scraper.classification.perplexity_client')
        logger.setLevel(logging.DEBUG)
        
        # Clear any existing log records
        caplog.clear()
        
        # First call - should miss cache and call API
        result1 = client.classify_pharmacy(SAMPLE_PHARMACY)
        
        # Convert to dict for comparison
        result1_dict = dict(result1._asdict()) if hasattr(result1, '_asdict') else dict(result1)
        expected_dict = dict(SAMPLE_RESPONSE_DICT)
        assert result1_dict == expected_dict, f"Expected {expected_dict}, got {result1_dict}"
        
        # Verify API was called once
        assert mock_make_api_call.call_count == 1, "API should be called once for first request"
        
        # Verify cache miss was logged
        assert any("Cache miss" in record.message for record in caplog.records), \
            "Expected 'Cache miss' log message not found"
        
        # Reset the mock and clear logs
        mock_make_api_call.reset_mock()
        caplog.clear()
        
        # Second call - should hit cache
        # Create a new instance to ensure we're not relying on in-memory caching
        with patch.object(client, 'cache') as mock_cache:
            mock_cache.get.return_value = api_response
            result2 = client.classify_pharmacy(SAMPLE_PHARMACY)
            
            # Convert to dict for comparison
            result2_dict = dict(result2._asdict()) if hasattr(result2, '_asdict') else dict(result2)
            assert result2_dict == expected_dict, f"Expected {expected_dict}, got {result2_dict}"
            
            # Verify cache.get was called
            mock_cache.get.assert_called_once()
            
            # Verify API was not called again
            mock_make_api_call.assert_not_called()

    def test_force_reclassification(self, client_fixture):
        """Test that setting `force_reclassification=True` bypasses the cache."""
        client, mock_make_api_call, _ = client_fixture
        
        # Create a simplified response dictionary that matches what the API would return
        api_response = {
            'is_chain': False,
            'is_compounding': True,
            'confidence': 0.95,
            'reason': 'Test pharmacy is independently owned',
            'source': 'perplexity',
            'method': 'llm'
        }
        mock_make_api_call.return_value = api_response

        # First call - should call API
        result1 = client.classify_pharmacy(SAMPLE_PHARMACY)
        result1_dict = dict(result1._asdict()) if hasattr(result1, '_asdict') else dict(result1)
        expected_dict = dict(SAMPLE_RESPONSE_DICT)
        assert result1_dict == expected_dict, f"Expected {expected_dict}, got {result1_dict}"
        mock_make_api_call.assert_called_once()

        # Reset mock call count
        mock_make_api_call.reset_mock()
        
        # Enable force reclassification
        client.force_reclassification = True
        
        # Second call - should call API again due to force_reclassification
        result2 = client.classify_pharmacy(SAMPLE_PHARMACY)
        result2_dict = dict(result2._asdict()) if hasattr(result2, '_asdict') else dict(result2)
        assert result2_dict == expected_dict, f"Expected {expected_dict}, got {result2_dict}"
        mock_make_api_call.assert_called_once()  # API should be called again


    def test_cache_disabled(self, caplog):
        """Test that the cache is not used when the client is initialized with cache_dir=None."""
        # Create a simplified response dictionary that matches what the API would return
        api_response = {
            'is_chain': False,
            'is_compounding': True,
            'confidence': 0.95,
            'reason': 'Test pharmacy is independently owned',
            'source': 'perplexity',
            'method': 'llm'
        }
        
        # Setup mock for the API call
        with patch('pharmacy_scraper.classification.perplexity_client.PerplexityClient._make_api_call', 
                 return_value=api_response) as mock_make_api_call:
            # Create client with cache disabled
            client = PerplexityClient(
                api_key="fake_key",
                cache_dir=None,  # Cache disabled
                force_reclassification=False
            )
            
            # First call - should call API
            result1 = client.classify_pharmacy(SAMPLE_PHARMACY)
            result1_dict = dict(result1._asdict()) if hasattr(result1, '_asdict') else dict(result1)
            expected_dict = dict(SAMPLE_RESPONSE_DICT)
            assert result1_dict == expected_dict, f"Expected {expected_dict}, got {result1_dict}"
            
            # Verify API was called once
            assert mock_make_api_call.call_count == 1, "API should be called for first request"
            
            # Reset mock call count
            mock_make_api_call.reset_mock()
            
            # Second call - should call API again since cache is disabled
            result2 = client.classify_pharmacy(SAMPLE_PHARMACY)
            result2_dict = dict(result2._asdict()) if hasattr(result2, '_asdict') else dict(result2)
            assert result2_dict == expected_dict, f"Expected {expected_dict}, got {result2_dict}"
            
            # API should be called again since cache is disabled
            assert mock_make_api_call.call_count == 1, "API should be called again when caching is disabled"

    @patch('pharmacy_scraper.classification.perplexity_client.Cache')
    def test_cache_directory_creation(self, MockCache, tmp_path):
        """Test that cache directory is properly created when cache is enabled."""
        # Reset mock to clear any previous calls
        MockCache.reset_mock()
        
        # Create a test cache directory
        test_cache_dir = tmp_path / "test_cache"
        
        # Create a mock cache instance
        mock_cache_instance = MagicMock()
        MockCache.return_value = mock_cache_instance
        
        # Create client with cache enabled
        with patch('pharmacy_scraper.classification.perplexity_client.Path.mkdir') as mock_mkdir:
            client = PerplexityClient(
                api_key="test_key",
                cache_dir=str(test_cache_dir),
                cache_ttl_seconds=None
            )
            
            # Verify Cache was initialized with the correct directory
            MockCache.assert_called_once_with(
                cache_dir=str(test_cache_dir),
                ttl=0,  # TTL is disabled in tests
                cleanup_interval=300
            )
            
            # Verify the cache directory was created
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
            
            # Verify the cache instance is stored
            assert client.cache is mock_cache_instance

    @patch('pharmacy_scraper.classification.perplexity_client.Cache.get')
    @patch('pharmacy_scraper.classification.perplexity_client.Cache.set')
    def test_cache_directory_creation(self, mock_cache_get, mock_cache_set, tmp_path):
        cache_dir = tmp_path / "new_cache_dir"
        assert not cache_dir.exists()
        # Create client with just the required parameters
        PerplexityClient(api_key="test_key", cache_dir=str(cache_dir))
        # Verify the cache directory was created
        assert cache_dir.exists()
