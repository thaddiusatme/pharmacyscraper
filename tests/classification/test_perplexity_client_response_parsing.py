"""
Tests for response parsing functionality in the PerplexityClient.

This module covers various response formats, edge cases, and the
correct parsing of responses from the Perplexity API.
"""
import json
import pytest
from unittest.mock import patch, MagicMock

from pharmacy_scraper.classification.perplexity_client import (
    PerplexityClient,
    ResponseParseError,
    ClassificationSource
)
from pharmacy_scraper.classification.models import PharmacyData


# Sample pharmacy data for testing
SAMPLE_PHARMACY = {
    'name': 'Test Pharmacy',
    'address': '123 Main St',
    'city': 'Test City',
    'state': 'CA',
    'zip': '12345',
    'phone': '(555) 123-4567',
    'categories': 'Pharmacy, Health',
    'website': 'https://testpharmacy.com',
    'is_chain': False
}

SAMPLE_PHARMACY_DATA = PharmacyData.from_dict(SAMPLE_PHARMACY)

# Helper function to create a mock response
def create_mock_response(content):
    class MockMessage:
        def __init__(self, content):
            self.content = content if isinstance(content, str) else json.dumps(content)
    
    class MockChoice:
        def __init__(self, content):
            self.message = MockMessage(content)
    
    class MockResponse:
        def __init__(self, content):
            self.choices = [MockChoice(content)]
    
    return MockResponse(content)


class TestPerplexityClientResponseParsing:
    """Tests for response parsing in the PerplexityClient."""
    
    @pytest.fixture
    def client(self):
        """Fixture to create a mocked client."""
        with patch('pharmacy_scraper.classification.perplexity_client.OpenAI'):
            client = PerplexityClient(api_key="test-key", model_name="test-model")
            # Don't mock _create_prompt in the fixture to allow real testing
            return client
    
    def test_parse_standard_json_response(self, client):
        """Test parsing a standard JSON response."""
        # Standard response with all fields
        response = create_mock_response({
            "classification": "independent",
            "is_chain": False,
            "is_compounding": True,
            "confidence": 0.95,
            "explanation": "This is an independent pharmacy."
        })
        
        result = client._parse_response(response, SAMPLE_PHARMACY_DATA)
        
        assert result.classification == "independent"
        assert result.is_chain is False
        assert result.is_compounding is True
        assert result.confidence == 0.95
        assert result.explanation == "This is an independent pharmacy."
        assert result.source == ClassificationSource.PERPLEXITY
        assert result.model == "test-model"
        assert result.pharmacy_data == SAMPLE_PHARMACY_DATA
    
    def test_parse_response_with_markdown_fences(self, client):
        """Test parsing a response with markdown code fences."""
        # Response wrapped in markdown code fences
        markdown_response = create_mock_response("""```json
{
  "classification": "independent",
  "is_chain": false,
  "is_compounding": true,
  "confidence": 0.95,
  "explanation": "This is an independent pharmacy."
}
```""")
        
        result = client._parse_response(markdown_response, SAMPLE_PHARMACY_DATA)
        
        assert result.classification == "independent"
        assert result.is_chain is False
        assert result.is_compounding is True
        assert result.confidence == 0.95
        assert result.explanation == "This is an independent pharmacy."
    
    def test_parse_response_with_missing_optional_fields(self, client):
        """Test parsing a response with missing optional fields."""
        # Response missing some optional fields
        minimal_response = create_mock_response({
            "classification": "independent",
            "is_chain": False
        })
        
        result = client._parse_response(minimal_response, SAMPLE_PHARMACY_DATA)
        
        assert result.classification == "independent"
        assert result.is_chain is False
        assert result.is_compounding is False  # Default value
        assert result.confidence == 0.0  # Default value
        assert result.explanation == ""  # Default value
    
    def test_parse_response_with_missing_required_fields(self, client):
        """Test parsing a response with missing required fields."""
        # Response missing required fields
        incomplete_response = create_mock_response({
            "confidence": 0.95,
            "explanation": "This is an independent pharmacy."
        })
        
        with pytest.raises(ResponseParseError, match="Missing required keys"):
            client._parse_response(incomplete_response, SAMPLE_PHARMACY_DATA)
    
    def test_parse_response_with_json_decode_error(self, client):
        """Test parsing a response with invalid JSON."""
        # Response with invalid JSON
        invalid_json = create_mock_response("This is not valid JSON")
        
        with pytest.raises(ResponseParseError, match="Invalid response format"):
            client._parse_response(invalid_json, SAMPLE_PHARMACY_DATA)
    
    def test_parse_response_with_empty_response(self, client):
        """Test parsing an empty response."""
        # Empty response
        empty_response = MagicMock()
        empty_response.choices = []
        
        with pytest.raises(ResponseParseError, match="Invalid response format"):
            client._parse_response(empty_response, SAMPLE_PHARMACY_DATA)
    
    def test_parse_response_with_none_content(self, client):
        """Test parsing a response with None content."""
        # Response with None content
        class MockMessage:
            def __init__(self):
                self.content = None
        
        class MockChoice:
            def __init__(self):
                self.message = MockMessage()
        
        class MockResponse:
            def __init__(self):
                self.choices = [MockChoice()]
        
        with pytest.raises(ResponseParseError, match="Invalid response format"):
            client._parse_response(MockResponse(), SAMPLE_PHARMACY_DATA)
    
    def test_parse_response_with_extra_fields(self, client):
        """Test parsing a response with extra fields that should be ignored."""
        # Response with extra fields
        response_with_extra = create_mock_response({
            "classification": "independent",
            "is_chain": False,
            "is_compounding": True,
            "confidence": 0.95,
            "explanation": "This is an independent pharmacy.",
            "extra_field": "should be ignored",
            "another_extra": 123
        })
        
        result = client._parse_response(response_with_extra, SAMPLE_PHARMACY_DATA)
        
        assert result.classification == "independent"
        assert result.is_chain is False
        assert result.is_compounding is True
        assert result.confidence == 0.95
        assert result.explanation == "This is an independent pharmacy."
        # Extra fields should be ignored
        with pytest.raises(AttributeError):
            assert result.extra_field is None
    
    def test_create_prompt_with_complete_data(self, client):
        """Test creating a prompt with complete pharmacy data."""
        # Don't mock the method - use the real implementation
        client._create_prompt = PerplexityClient._create_prompt.__get__(client)
        prompt = client._create_prompt(SAMPLE_PHARMACY_DATA)
        
        # Verify all fields are included
        assert "Test Pharmacy" in prompt
        assert "123 Main St" in prompt
        assert "Pharmacy, Health" in prompt
        assert "https://testpharmacy.com" in prompt
        assert "JSON" in prompt  # Check for JSON instruction
        
    def test_create_prompt_with_minimal_data(self, client):
        """Test creating a prompt with minimal pharmacy data."""
        # Don't mock the method - use the real implementation
        client._create_prompt = PerplexityClient._create_prompt.__get__(client)
        minimal_data = PharmacyData.from_dict({"name": "Minimal Pharmacy"})
        prompt = client._create_prompt(minimal_data)
        
        # Verify required fields and None for missing
        assert "Minimal Pharmacy" in prompt
        assert "None" in prompt  # None values for missing fields
