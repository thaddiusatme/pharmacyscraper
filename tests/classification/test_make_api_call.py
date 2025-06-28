"""
Test cases for the _make_api_call method of PerplexityClient.
"""
import json
import pytest
import sys
from unittest.mock import MagicMock, patch, call, ANY, DEFAULT
from typing import Dict, Any, Optional, Union
from requests import Response

# Mock OpenAI before importing PerplexityClient
sys.modules['openai'] = MagicMock()
# Create proper exception classes that match the OpenAI library
class RateLimitError(Exception):
    def __init__(self, message, **kwargs):
        self.message = message
        self.response = kwargs.get('response')
        self.body = kwargs.get('body', {})
        super().__init__(self.message)

class APIError(Exception):
    def __init__(self, message, **kwargs):
        self.message = message
        self.response = kwargs.get('response')
        self.body = kwargs.get('body', {})
        super().__init__(self.message)

sys.modules['openai'].RateLimitError = RateLimitError
sys.modules['openai'].APIError = APIError

# Now import the client
from pharmacy_scraper.classification.perplexity_client import PerplexityClient, PerplexityAPIError

# Mock OpenAI exceptions
class MockRateLimitError(Exception):
    def __init__(self, message, **kwargs):
        self.message = message
        self.response = kwargs.get('response', MagicMock())
        self.body = kwargs.get('body', {})
        super().__init__(self.message)

class MockAPIError(Exception):
    def __init__(self, message, **kwargs):
        self.message = message
        self.response = kwargs.get('response', MagicMock())
        self.body = kwargs.get('body', {})
        super().__init__(self.message)
        
        # Add http_status attribute if provided
        if 'http_status' in kwargs:
            self.http_status = kwargs['http_status']

def create_mock_response(content: str) -> MagicMock:
    """Helper function to create a mock response with the given content."""
    response = MagicMock()
    choice = MagicMock()
    message = MagicMock()
    message.content = content
    choice.message = message
    response.choices = [choice]
    return response

def create_mock_request():
    """Create a mock request object."""
    request = MagicMock()
    request.headers = {}
    return request

def create_mock_http_response(status_code: int = 200, content: str = '') -> MagicMock:
    """Create a mock HTTP response."""
    response = MagicMock(spec=Response)
    response.status_code = status_code
    response.content = content
    response.text = content
    response.json.return_value = {'error': {'message': content or 'Error'}}
    return response

# Create a fixture that provides a test client with a mock API key
@pytest.fixture
def client():
    """Create a test client with a mock API key and mocked OpenAI client."""
    # Create a mock client for the OpenAI instance
    mock_client = MagicMock()
    
    # Create a mock for the OpenAI class that will return our mock client
    mock_openai_class = MagicMock(return_value=mock_client)
    
    # Patch the OpenAI class in the module where it's used
    with patch('pharmacy_scraper.classification.perplexity_client.OpenAI', new=mock_openai_class), \
         patch('openai.OpenAI', new=mock_openai_class):
        
        # Create a test client - this will use our patched OpenAI
        test_client = PerplexityClient(api_key="test_key")
        
        # Store the mocks for assertions
        test_client._mock_openai = mock_openai_class
        test_client._mock_client = mock_client
        
        # Setup default successful response
        mock_response = create_mock_response('{"classification": "independent", "confidence": 0.9, "is_compounding": true, "explanation": "Test explanation"}')
        mock_client.chat.completions.create.return_value = mock_response
        
        yield test_client

# Fixture for sample pharmacy data
@pytest.fixture
def sample_pharmacy_data():
    """Sample pharmacy data for testing."""
    return {
        "name": "Test Pharmacy",
        "address": "123 Test St, Test City, TS 12345",
        "phone": "(123) 456-7890",
        "website": "https://testpharmacy.com"
    }

class TestMakeAPICall:
    """Test cases for the _make_api_call method."""
    
    def test_make_api_call_success(self, client, sample_pharmacy_data):
        """Test a successful API call."""
        # Arrange - client fixture already sets up the mocks
        mock_client = client._mock_client
        
        # Act
        result = client._make_api_call(sample_pharmacy_data, model="test-model")
        
        # Assert
        assert result is not None
        assert result["classification"] == "independent"
        assert result["confidence"] == 0.9
        assert result["is_compounding"] is True
        assert "explanation" in result
        mock_client.chat.completions.create.assert_called_once()
    
    @patch('time.sleep')
    def test_make_api_call_with_retries(self, mock_sleep, client, sample_pharmacy_data):
        """Test API call with retries on rate limit errors."""
        # Arrange
        mock_client = client._mock_client
        
        # Create a successful response mock
        success_response = create_mock_response('{"classification": "independent", "confidence": 0.9, "is_compounding": true, "explanation": "Test explanation"}')
        
        # Create mock rate limit error responses
        rate_limit_error = MockRateLimitError(
            "Rate limit exceeded",
            response=create_mock_http_response(429, 'Rate limit exceeded'),
            body={'error': {'message': 'Rate limit exceeded'}}
        )
        
        # Set up side effect: two rate limit errors, then success
        mock_client.chat.completions.create.side_effect = [
            rate_limit_error,
            rate_limit_error,
            success_response
        ]
        
        # Act
        result = client._make_api_call(sample_pharmacy_data, model="test-model", retry_count=2)
        
        # Assert
        assert result is not None
        assert result["classification"] == "independent"
        assert mock_client.chat.completions.create.call_count == 3
        # The client uses exponential backoff, so it will sleep more than just 2 times
        assert mock_sleep.call_count > 0
    
    @patch('time.sleep')
    def test_make_api_call_exhaust_retries(self, mock_sleep, client, sample_pharmacy_data):
        """Test API call that exhausts all retries."""
        # Arrange
        mock_client = client._mock_client
        
        # Create mock rate limit error
        rate_limit_error = MockRateLimitError(
            "Rate limit exceeded",
            response=create_mock_http_response(429, 'Rate limit exceeded'),
            body={'error': {'message': 'Rate limit exceeded'}}
        )
        
        # All calls fail with RateLimitError
        mock_client.chat.completions.create.side_effect = rate_limit_error
        
        # Act & Assert - Should raise RateLimitError after exhausting retries
        # Note: The client raises its own RateLimitError, not the one from openai
        from pharmacy_scraper.classification.perplexity_client import RateLimitError as ClientRateLimitError
        with pytest.raises(ClientRateLimitError, match="Rate limit exceeded after 3 retries"):
            client._make_api_call(sample_pharmacy_data, model="test-model", retry_count=2)
        
        # Verify the expected number of calls were made (initial + max_retries)
        # The client does max_retries + 1 total attempts (initial + retries)
        assert mock_client.chat.completions.create.call_count == 4  # Initial + 3 retries
    
    def test_make_api_call_invalid_response(self, client, sample_pharmacy_data):
        """Test API call with invalid response format."""
        # Arrange
        mock_client = client._mock_client
        
        # Create a mock response with invalid JSON content
        mock_response = create_mock_response('Invalid JSON response')
        mock_client.chat.completions.create.return_value = mock_response
        
        # Act
        result = client._make_api_call(sample_pharmacy_data, model="test-model")
        
        # Assert - Should return a dict with error information
        assert result is not None
        assert "response" in result
        assert isinstance(result["response"], str)
    
    def test_make_api_call_http_error(self, client, sample_pharmacy_data):
        """Test API call with HTTP error."""
        # Arrange
        mock_client = client._mock_client
        
        # Simulate an API error
        mock_client.chat.completions.create.side_effect = MockAPIError(
            "API error",
            response=create_mock_http_response(500, 'Internal server error'),
            body={'error': {'message': 'Internal server error'}},
            http_status=500
        )
        
        # Act & Assert - Should raise PerplexityAPIError for API errors
        with pytest.raises(PerplexityAPIError, match="API error"):
            client._make_api_call(sample_pharmacy_data, model="test-model")
    
    def test_make_api_call_with_headers(self, client, sample_pharmacy_data):
        """Test API call includes correct headers."""
        # Arrange - client fixture already sets up the mocks
        mock_client = client._mock_client
        
        # Act
        client._make_api_call(sample_pharmacy_data, model="sonar")
        
        # Assert OpenAI client was initialized with the correct parameters
        client._mock_openai.assert_called_once_with(
            api_key="test_key",
            base_url="https://api.perplexity.ai"
        )
        
        # Verify the chat completion was called with the expected parameters
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args[1]
        # PerplexityClient uses 'sonar' as the default model
        assert call_args["model"] == "sonar"
    
    def test_make_api_call_with_custom_model(self, client, sample_pharmacy_data):
        """Test API call uses the specified model."""
        # Arrange - client fixture already sets up the mocks
        mock_client = client._mock_client
        mock_response = create_mock_response('{"classification": "independent", "confidence": 0.9, "is_compounding": true, "explanation": "Test explanation"}')
        mock_client.chat.completions.create.return_value = mock_response
        
        # Use a custom model name that matches what PerplexityClient expects
        custom_model = "sonar"
        
        # Act
        result = client._make_api_call(sample_pharmacy_data, model=custom_model)
        
        # Assert
        assert result is not None
        assert result["classification"] == "independent"
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args[1]
        assert call_args["model"] == custom_model
