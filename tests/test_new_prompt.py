"""Test the new prompt format and response parsing."""
import os
import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from pharmacy_scraper.classification.perplexity_client import PerplexityClient, _generate_cache_key

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Mock response for testing
MOCK_RESPONSE = """```json
{
    "classification": "independent",
    "is_compounding": true,
    "confidence": 0.92,
    "explanation": "This is an independent pharmacy because it's locally owned and not part of a major chain. The business name and services indicate it's a compounding pharmacy."
}
```"""

def test_prompt_generation():
    """Test that the prompt is generated correctly with few-shot examples."""
    client = PerplexityClient(api_key="test_key")
    pharmacy_data = {
        "name": "Test Pharmacy",
        "address": "123 Test St, Test City, TS 12345",
        "categoryName": "Pharmacy, Health & Medical",
        "website": "testpharmacy.com"
    }
    
    prompt = client._generate_prompt(pharmacy_data)
    
    # Check that the prompt includes our examples
    assert "EXAMPLES" in prompt
    assert "Main Street Pharmacy" in prompt
    assert "CVS Pharmacy" in prompt
    assert "Union Avenue Compounding Pharmacy" in prompt
    
    # Check that the current pharmacy data is in the prompt
    assert "Test Pharmacy" in prompt
    assert "testpharmacy.com" in prompt
    
    # Check that the instructions are clear
    assert "Respond with a single JSON object containing" in prompt
    assert "explanation: A brief explanation of your reasoning" in prompt

def test_response_parsing():
    """Test parsing of the response with code blocks and explanation."""
    client = PerplexityClient(api_key="test_key")
    
    # Create a mock response object
    mock_choice = MagicMock()
    mock_choice.message.content = MOCK_RESPONSE
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    
    # Test parsing
    result = client._parse_response(mock_response)
    
    # Check the parsed data
    assert result is not None
    assert result["classification"] == "independent"
    assert result["is_compounding"] is True
    assert 0 <= result["confidence"] <= 1
    assert "explanation" in result
    assert "locally owned" in result["explanation"]

@patch('pharmacy_scraper.classification.perplexity_client.OpenAI')
def test_integration(mock_openai):
    """Test the full classification flow with a mock API response."""
    # Setup mock response with the expected structure
    mock_choice = MagicMock()
    mock_choice.message.content = MOCK_RESPONSE
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    
    # Setup mock client
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai.return_value = mock_client
    
    # Create a temporary directory for cache
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create client with mocked API and temp cache
        client = PerplexityClient(api_key="test_key", cache_dir=temp_dir)
        
        # Test classification data
        pharmacy_data = {
            "name": "Test Pharmacy",
            "address": "123 Test St, Test City, TS 12345",
            "categoryName": "Pharmacy, Health & Medical",
            "website": "testpharmacy.com"
        }
        
        # Test classification
        result = client.classify_pharmacy(pharmacy_data)
        
        # Verify the API was called with the right parameters
        mock_openai.assert_called_once_with(api_key="test_key", base_url="https://api.perplexity.ai")
        mock_client.chat.completions.create.assert_called_once()
        
        # Verify the result
        assert result is not None, "Result should not be None"
        assert "classification" in result, f"Result should contain 'classification' key. Got: {result}"
        assert result["classification"] == "independent", f"Expected 'independent' but got {result['classification']}"
        assert "is_compounding" in result, f"Result should contain 'is_compounding' key. Got: {result}"
        assert result["is_compounding"] is True, f"Expected is_compounding to be True but got {result['is_compounding']}"
        assert "explanation" in result, f"Result should contain 'explanation' key. Got: {result}"
        assert "confidence" in result, f"Result should contain 'confidence' key. Got: {result}"
        assert 0 <= result["confidence"] <= 1, f"Confidence should be between 0 and 1, got {result['confidence']}"
        
        # Verify the result was cached
        cache_key = _generate_cache_key(pharmacy_data, client.model_name)
        cache_file = Path(temp_dir) / f"{cache_key}.json"
        assert cache_file.exists(), "Result was not cached"
