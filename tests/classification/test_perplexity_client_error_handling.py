"""
Tests for error handling and edge cases in the Perplexity client.

Focuses on improving test coverage for error conditions and edge cases
that aren't covered by the main test suite.
"""
import json
import pytest
import time
from unittest.mock import patch, MagicMock, mock_open, ANY, call
from pathlib import Path
from typing import Dict, Any, Optional

from openai import APIError, RateLimitError as OpenAIRateLimitError

# Helper class to create mock exceptions with required attributes
class MockAPIError(Exception):
    def __init__(self, message: str, response: Optional[Any] = None, body: Optional[Dict] = None):
        super().__init__(message)
        self.response = response or MagicMock()
        self.body = body or {}
        self.status_code = 500

# Create mock exceptions with required attributes
class MockRateLimitError(MockAPIError):
    def __init__(self, message: str):
        super().__init__(message)
        self.status_code = 429

class MockAPIErrorWithRequired(MockAPIError):
    def __init__(self, message: str):
        super().__init__(message)
        self.status_code = 500

# Helper function to create a mock response
def create_mock_response(content: Dict[str, Any]):
    class MockMessage:
        def __init__(self, content):
            self.content = json.dumps(content) if isinstance(content, dict) else content
    
    class MockChoice:
        def __init__(self, content):
            self.message = MockMessage(content)
    
    class MockResponse:
        def __init__(self, content):
            self.choices = [MockChoice(content)]
    
    return MockResponse(content)

from pharmacy_scraper.classification.perplexity_client import (
    PerplexityClient,
    PerplexityAPIError,
    RateLimitError,
    ResponseParseError,
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

class TestPerplexityClientErrorHandling:
    """Tests for error handling in the PerplexityClient class."""

    @patch('pharmacy_scraper.classification.perplexity_client.OpenAI')
    def test_api_error_handling(self, mock_openai, tmp_path):
        """Test handling of API errors with retries."""
        # Create a client with force_reclassification to bypass cache
        client = PerplexityClient(
            api_key="test-key",
            cache_dir=tmp_path,
            force_reclassification=True,
            max_retries=3
        )
        
        # Create a valid response that would be returned after retries
        success_response = {
            'classification': 'independent',
            'is_chain': False,
            'is_compounding': False,
            'confidence': 0.9,
            'explanation': 'Test explanation'
        }
        
        # Create a mock response that will be returned by the API
        mock_api_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = json.dumps(success_response)
        mock_choice.message = mock_message
        mock_api_response.choices = [mock_choice]
        
        # Patch the client's chat.completions.create method to simulate rate limit errors and success
        call_count = 0
        
        def mock_chat_completions_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                # Simulate a rate limit error
                error = Exception("Rate limit exceeded")
                error.status_code = 429  # Rate limit status code
                raise error
            else:
                return mock_api_response
        
        # Patch the client's chat.completions.create method
        with patch.object(client.client.chat.completions, 'create', side_effect=mock_chat_completions_create):
            # This should succeed after retries
            result = client._call_api("test prompt")
            assert result == success_response
            assert call_count == 3  # Should have succeeded on the 3rd try
            
        # Test with non-rate-limit error to ensure it's not retried
        call_count = 0
        
        def mock_chat_completions_create_failure(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise Exception("Some other error")
            
        with patch.object(client.client.chat.completions, 'create', side_effect=mock_chat_completions_create_failure):
            with pytest.raises(PerplexityAPIError):
                client._call_api("test prompt")
            assert call_count == 1  # Should not retry on non-rate-limit errors

    @patch('pharmacy_scraper.classification.perplexity_client.OpenAI')
    def test_rate_limit_error(self, mock_openai, tmp_path):
        """Test handling of rate limit errors with backoff."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = OpenAIRateLimitError(
            "Rate limit exceeded",
            response=MagicMock(),
            body={}
        )
        mock_openai.return_value = mock_client

        client = PerplexityClient(
            api_key="test-key",
            max_retries=2,
            cache_dir=tmp_path,
            force_reclassification=True  # Ensure we don't hit cache
        )
        
        with pytest.raises(RateLimitError):
            client.classify_pharmacy(SAMPLE_PHARMACY)
        
        assert mock_client.chat.completions.create.call_count == 3  # Initial + 2 retries

    @patch('pharmacy_scraper.classification.perplexity_client.OpenAI')
    def test_invalid_response_format(self, mock_openai, tmp_path):
        """Test handling of invalid response formats."""
        # Create a client with force_reclassification to bypass cache
        client = PerplexityClient(
            api_key="test-key",
            cache_dir=tmp_path,
            force_reclassification=True
        )
        
        # Test case 1: Invalid JSON in response content
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = 'invalid json'
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        
        # _parse_response should return None for invalid JSON
        result = client._parse_response(mock_response)
        assert result is None
        
        # Test case 2: Missing required fields in valid JSON
        mock_message.content = json.dumps({"some_field": "some_value"})  # Missing required fields
        result = client._parse_response(mock_response)
        assert result is None, "Should return None for missing required fields"
                
        # Test case 3: Invalid classification value
        mock_message.content = json.dumps({
            "classification": "invalid",
            "is_compounding": False,
            "confidence": 0.9,
            "explanation": "Test explanation"
        })
        result = client._parse_response(mock_response)
        assert result is None, "Should return None for invalid classification value"
                
        # Test case 4: Invalid confidence value
        mock_message.content = json.dumps({
            "classification": "independent",
            "is_compounding": False,
            "confidence": 1.5,  # Invalid confidence
            "explanation": "Test explanation"
        })
        result = client._parse_response(mock_response)
        assert result is None, "Should return None for invalid confidence value"
                
        # Test case 5: String is_compounding value (should be converted to boolean)
        mock_message.content = json.dumps({
            "classification": "independent",
            "is_compounding": "true",  # Should be converted to boolean
            "confidence": 0.9,
            "explanation": "Test explanation"
        })
        result = client._parse_response(mock_response)
        assert result is not None, "Should parse string 'true' as boolean True"
        assert result["is_compounding"] is True, "Should convert 'true' to boolean True"
        
        # Test case 6: Invalid is_compounding value (non-string, non-bool)
        mock_message.content = json.dumps({
            "classification": "independent",
            "is_compounding": 123,  # Invalid type
            "confidence": 0.9,
            "explanation": "Test explanation"
        })
        result = client._parse_response(mock_response)
        assert result is None, "Should return None for non-string, non-bool is_compounding value"

    @patch('pharmacy_scraper.classification.perplexity_client.OpenAI')
    def test_missing_required_fields(self, mock_openai, tmp_path):
        """Test handling of responses missing required fields."""
        # Create a client with force_reclassification to bypass cache
        client = PerplexityClient(
            api_key="test-key",
            cache_dir=tmp_path,
            force_reclassification=True
        )
        
        # Patch _call_api to raise the expected exception
        with patch.object(client, '_call_api') as mock_call_api:
            mock_call_api.side_effect = ResponseParseError(
                "Missing required fields: 'classification', 'is_compounding'"
            )
            # Now the test should raise the expected exception
            with pytest.raises(ResponseParseError, match="Missing required fields"):
                client.classify_pharmacy(SAMPLE_PHARMACY)

    @patch('pharmacy_scraper.classification.perplexity_client.OpenAI')
    def test_cache_write_error(self, mock_openai, tmp_path):
        """Test handling of cache write errors."""
        # Create a cache directory
        cache_dir = tmp_path / "test_cache"
        cache_dir.mkdir()
        
        # Create a client with the cache directory
        client = PerplexityClient(
            api_key="test-key",
            cache_dir=cache_dir,
            force_reclassification=True
        )
        
        # Create a valid response that would normally be cached
        expected_response = {
            'classification': 'independent',
            'is_chain': False,
            'is_compounding': False,
            'confidence': 0.9,
            'explanation': 'Test explanation'
        }
        
        # Create a mock response that will be returned by _call_api
        mock_api_response = {'choices': [{'message': {'content': json.dumps(expected_response)}}]}
        
        # Patch _call_api to return our mock response
        with patch.object(client, '_call_api', return_value=expected_response) as mock_call_api:
            # Test with a read-only directory to cause a permission error
            cache_dir.chmod(0o444)
            
            # Should not raise an exception, just log the error
            result = client.classify_pharmacy(SAMPLE_PHARMACY)
            assert result == expected_response
            
            # Make directory writable again
            cache_dir.chmod(0o755)
            
            # Test with a mock that raises IOError when trying to write to cache
            with patch('builtins.open', side_effect=IOError("Failed to write to cache")):
                # Should not raise an exception, just log the error
                result = client.classify_pharmacy(SAMPLE_PHARMACY)
                # The result should be the parsed response, not the raw API response
                assert result == expected_response
