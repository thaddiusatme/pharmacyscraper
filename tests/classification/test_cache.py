"""Tests for the cache functionality in PerplexityClient."""
import json
import os
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from pharmacy_scraper.classification.perplexity_client import PerplexityClient, _generate_cache_key

# Sample pharmacy data for testing
SAMPLE_PHARMACY = {
    "title": "Test Pharmacy",
    "address": "123 Test St, Test City, TS 12345",
    "phone": "(555) 123-4567"
}

# Sample API response
SAMPLE_RESPONSE = {
    "is_chain": False,
    "is_compounding": True,
    "confidence": 0.95,
    "explanation": "This is a test response"
}

@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary cache directory for testing."""
    cache_dir = tmp_path / "test_cache"
    cache_dir.mkdir()
    yield str(cache_dir)
    shutil.rmtree(cache_dir, ignore_errors=True)

@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    
    mock_message.content = json.dumps(SAMPLE_RESPONSE)
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
    
    return mock_client

def test_cache_key_generation():
    """Test that cache keys are generated consistently."""
    # Same data should produce same key
    key1 = _generate_cache_key(SAMPLE_PHARMACY, "test-model")
    key2 = _generate_cache_key(SAMPLE_PHARMACY, "test-model")
    assert key1 == key2
    
    # Different data should produce different keys
    different_pharmacy = SAMPLE_PHARMACY.copy()
    different_pharmacy["title"] = "Different Pharmacy"
    key3 = _generate_cache_key(different_pharmacy, "test-model")
    assert key1 != key3
    
    # Different models should produce different keys
    key4 = _generate_cache_key(SAMPLE_PHARMACY, "different-model")
    assert key1 != key4

def test_cache_read_hit(temp_cache_dir, mock_openai_client):
    """Test reading from cache when a cache hit occurs."""
    # Create a cached file
    client = PerplexityClient(
        api_key="test_key",
        cache_dir=temp_cache_dir,
        openai_client=mock_openai_client
    )
    
    # First call - should call the API and cache the result
    result1 = client.classify_pharmacy(SAMPLE_PHARMACY)
    assert mock_openai_client.chat.completions.create.called
    
    # Reset the mock to track subsequent calls
    mock_openai_client.chat.completions.create.reset_mock()
    
    # Second call with same data - should use cache
    result2 = client.classify_pharmacy(SAMPLE_PHARMACY)
    assert not mock_openai_client.chat.completions.create.called
    assert result1 == result2

def test_cache_read_miss(temp_cache_dir, mock_openai_client):
    """Test cache miss behavior."""
    client = PerplexityClient(
        api_key="test_key",
        cache_dir=temp_cache_dir,
        openai_client=mock_openai_client
    )
    
    # Call with data not in cache - should call API
    result = client.classify_pharmacy(SAMPLE_PHARMACY)
    assert mock_openai_client.chat.completions.create.called
    assert result is not None

def test_force_reclassification(temp_cache_dir, mock_openai_client):
    """Test that force_reclassification bypasses the cache."""
    client = PerplexityClient(
        api_key="test_key",
        cache_dir=temp_cache_dir,
        openai_client=mock_openai_client,
        force_reclassification=True
    )
    
    # First call - should call API due to force_reclassification
    result1 = client.classify_pharmacy(SAMPLE_PHARMACY)
    assert mock_openai_client.chat.completions.create.called
    
    # Reset the mock
    mock_openai_client.chat.completions.create.reset_mock()
    
    # Second call - should still call API due to force_reclassification
    result2 = client.classify_pharmacy(SAMPLE_PHARMACY)
    assert mock_openai_client.chat.completions.create.called

def test_cache_write_failure(temp_cache_dir, mock_openai_client, caplog, monkeypatch):
    """Test behavior when cache write fails."""
    # Create a mock for the cache file operations
    mock_file = MagicMock()
    mock_file.__enter__.return_value = mock_file
    
    def mock_open(*args, **kwargs):
        if 'w' in args[1]:  # Only fail on write operations
            raise IOError("Simulated write error")
        return MagicMock()
    
    # Apply the mock
    monkeypatch.setattr("builtins.open", mock_open)
    
    client = PerplexityClient(
        api_key="test_key",
        cache_dir=temp_cache_dir,
        openai_client=mock_openai_client
    )
    
    # Should still work, just log the error
    result = client.classify_pharmacy(SAMPLE_PHARMACY)
    
    # Verify the error was logged
    assert any("Failed to write to cache file" in record.message for record in caplog.records)
    assert result is not None

def test_cache_read_failure(temp_cache_dir, mock_openai_client, caplog):
    """Test behavior when cache read fails."""
    # Create a corrupted cache file
    cache_key = _generate_cache_key(SAMPLE_PHARMACY, "sonar")
    cache_file = Path(temp_cache_dir) / f"{cache_key}.json"
    with open(cache_file, 'w') as f:
        f.write("invalid json")
    
    client = PerplexityClient(
        api_key="test_key",
        cache_dir=temp_cache_dir,
        openai_client=mock_openai_client
    )
    
    # Should fall back to API call
    result = client.classify_pharmacy(SAMPLE_PHARMACY)
    assert "Failed to read cache file" in caplog.text
    assert mock_openai_client.chat.completions.create.called
    assert result is not None

def test_cache_disabled(mock_openai_client):
    """Test behavior when cache is disabled."""
    client = PerplexityClient(
        api_key="test_key",
        cache_dir=None,  # Disable cache
        openai_client=mock_openai_client
    )
    
    # First call - should call API
    result1 = client.classify_pharmacy(SAMPLE_PHARMACY)
    assert mock_openai_client.chat.completions.create.called
    
    # Reset the mock
    mock_openai_client.chat.completions.create.reset_mock()
    
    # Second call - should call API again (no caching)
    result2 = client.classify_pharmacy(SAMPLE_PHARMACY)
    assert mock_openai_client.chat.completions.create.called

def test_cache_directory_creation(temp_cache_dir):
    """Test that cache directory is created if it doesn't exist."""
    # Use a non-existent subdirectory
    cache_dir = os.path.join(temp_cache_dir, "nonexistent")
    
    # Should create the directory
    client = PerplexityClient(
        api_key="test_key",
        cache_dir=cache_dir
    )
    
    assert os.path.isdir(cache_dir)
