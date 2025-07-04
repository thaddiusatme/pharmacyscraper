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
        "categories": "Pharmacy, Health & Medical",
        "website": "testpharmacy.com"
    }
    
    from pharmacy_scraper.classification.models import PharmacyData
    pharmacy_obj = PharmacyData.from_dict(pharmacy_data)
    prompt = client._create_prompt(pharmacy_obj)
    
    # Check that the prompt includes the pharmacy data
    assert "Test Pharmacy" in prompt
    assert "123 Test St" in prompt
    assert "testpharmacy.com" in prompt
    
    # Check that the current pharmacy data is in the prompt
    assert "Test Pharmacy" in prompt
    assert "testpharmacy.com" in prompt
    
    # Check that the instructions are clear
    assert "Please respond with a JSON object containing" in prompt
    assert "explanation" in prompt

def test_response_parsing():
    """Test parsing of the response with code blocks and explanation."""
    client = PerplexityClient(api_key="test_key")
    
    # Create a mock response object
    mock_choice = MagicMock()
    mock_choice.message.content = MOCK_RESPONSE
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    
    # Create pharmacy data object for the test
    from pharmacy_scraper.classification.models import PharmacyData
    pharmacy_data = PharmacyData(
        name="Test Pharmacy",
        address="123 Test St, Test City, TS 12345",
        categories="Pharmacy, Health & Medical",
        website="testpharmacy.com"
    )
    
    # Test parsing
    result = client._parse_response(mock_response, pharmacy_data)
    
    # Check the parsed data
    assert result is not None
    assert result.classification == "independent"
    assert result.is_compounding is True
    assert 0 <= result.confidence <= 1
    assert result.explanation is not None
    assert "locally owned" in result.explanation

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
        # Create client with mocked API and temp cache - specify cache_ttl_seconds to avoid NoneType error
        client = PerplexityClient(api_key="test_key", cache_dir=temp_dir, cache_ttl_seconds=3600)
        
        # Test classification data
        pharmacy_data = {
            "name": "Test Pharmacy",
            "address": "123 Test St, Test City, TS 12345",
            "categories": "Pharmacy, Health & Medical",
            "website": "testpharmacy.com"
        }
        
        # Test classification
        result = client.classify_pharmacy(pharmacy_data)
        
        # Verify the API was called with the right parameters
        mock_openai.assert_called_once_with(api_key="test_key", base_url="https://api.perplexity.ai")
        mock_client.chat.completions.create.assert_called_once()
        
        # Verify the result - now using ClassificationResult object
        assert result is not None, "Result should not be None"
        assert hasattr(result, 'classification'), f"Result should have 'classification' attribute. Got: {result}"
        assert result.classification == "independent", f"Expected 'independent' but got {result.classification}"
        assert hasattr(result, 'is_compounding'), f"Result should have 'is_compounding' attribute. Got: {result}"
        assert result.is_compounding is True, f"Expected is_compounding to be True but got {result.is_compounding}"
        assert hasattr(result, 'explanation'), f"Result should have 'explanation' attribute. Got: {result}"
        assert hasattr(result, 'confidence'), f"Result should have 'confidence' attribute. Got: {result}"
        assert 0 <= result.confidence <= 1, f"Confidence should be between 0 and 1, got {result.confidence}"
        
        # Verify the result was cached
        from pharmacy_scraper.classification.models import PharmacyData
        pharmacy_obj = PharmacyData.from_dict(pharmacy_data)
        cache_key = _generate_cache_key(pharmacy_data, client.model_name)
        cache_file = Path(temp_dir) / f"{cache_key}.json"
        assert cache_file.exists(), "Result was not cached"
