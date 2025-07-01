"""
Test cases for the _make_api_call method of PerplexityClient.
"""
import pytest
from unittest.mock import MagicMock, patch
from requests import Response
import openai

from pharmacy_scraper.classification.perplexity_client import (
    PerplexityClient,
    PerplexityAPIError,
    ResponseParseError,
    RateLimitError as ClientRateLimitError,
)


def create_mock_response(content: str) -> MagicMock:
    """Helper function to create a mock response with the given content."""
    response = MagicMock()
    choice = MagicMock()
    message = MagicMock()
    message.content = content
    choice.message = message
    response.choices = [choice]
    return response


def create_mock_http_response(status_code: int = 200, content: str = '') -> MagicMock:
    """Create a mock HTTP response that mimics httpx.Response."""
    response = MagicMock(spec=Response)
    response.status_code = status_code
    response.content = content.encode()
    response.text = content
    response.headers = MagicMock()
    response.headers.get.return_value = "dummy-request-id"
    response.request = MagicMock()
    response.json.return_value = {'error': {'message': content or 'Error'}}
    return response


@pytest.fixture
def client():
    """Create a test client with a mocked OpenAI client."""
    with patch("pharmacy_scraper.classification.perplexity_client.OpenAI") as mock_openai_class:
        mock_client_instance = MagicMock()
        mock_openai_class.return_value = mock_client_instance

        test_client = PerplexityClient(
            api_key="test_key",
            cache_enabled=False,  # Disable caching for tests
        )
        # Patch the client directly since we can't pass it in constructor anymore
        test_client.client = mock_client_instance

        success_content = '{"classification": "independent", "confidence": 0.9, "is_compounding": true, "explanation": "Test explanation"}'
        mock_response = create_mock_response(success_content)
        mock_client_instance.chat.completions.create.return_value = mock_response

        yield test_client


@pytest.fixture
def sample_pharmacy_data():
    """Sample pharmacy data for testing."""
    return {
        "name": "Test Pharmacy",
        "address": "123 Test St, Test City, TS 12345",
    }


class TestMakeAPICall:
    """Test cases for the _make_api_call method."""

    def test_make_api_call_success(self, client, sample_pharmacy_data):
        """Test a successful API call."""
        result = client._make_api_call(sample_pharmacy_data, model="test-model")

        assert result is not None
        assert result["classification"] == "independent"
        client._mock_client.chat.completions.create.assert_called_once()

    @patch('time.sleep')
    def test_make_api_call_with_retries(self, mock_sleep, client, sample_pharmacy_data):
        """Test API call with retries on rate limit errors."""
        mock_client = client._mock_client

        success_response = create_mock_response(
            '{"classification": "independent", "confidence": 0.9, "is_compounding": true, "explanation": "Test explanation"}'
        )

        rate_limit_error = openai.RateLimitError(
            message="Rate limit exceeded",
            response=create_mock_http_response(429, 'Rate limit exceeded'),
            body={'error': {'message': 'Rate limit exceeded'}}
        )

        mock_client.chat.completions.create.side_effect = [
            rate_limit_error,
            rate_limit_error,
            success_response
        ]

        result = client._make_api_call(sample_pharmacy_data, model="test-model")

        assert result is not None
        assert result["classification"] == "independent"
        assert mock_client.chat.completions.create.call_count == 3
        assert mock_sleep.call_count == 2

    @patch('time.sleep')
    def test_make_api_call_exhaust_retries(self, mock_sleep, client, sample_pharmacy_data):
        """Test API call that exhausts all retries."""
        mock_client = client._mock_client

        rate_limit_error = openai.RateLimitError(
            message="Rate limit exceeded",
            response=create_mock_http_response(429, 'Rate limit exceeded'),
            body={'error': {'message': 'Rate limit exceeded'}}
        )

        mock_client.chat.completions.create.side_effect = rate_limit_error

        with pytest.raises(ClientRateLimitError):
            client._make_api_call(sample_pharmacy_data, model="test-model")

        assert mock_client.chat.completions.create.call_count == client.max_retries + 1

    def test_make_api_call_invalid_response(self, client, sample_pharmacy_data):
        """Test API call with invalid response format."""
        mock_client = client._mock_client
        mock_response = create_mock_response('Invalid JSON response')
        mock_client.chat.completions.create.return_value = mock_response

        with pytest.raises(ResponseParseError):
            client._make_api_call(sample_pharmacy_data, model="test-model")

    def test_api_error_is_propagated(self, client, sample_pharmacy_data):
        """Test that a generic APIError is wrapped and raised correctly."""
        mock_client = client._mock_client

        mock_http_resp = create_mock_http_response(500, 'Internal server error')
        api_error = openai.APIError(
            message="Internal server error",
            request=mock_http_resp.request,
            body={'error': {'message': 'Internal server error'}}
        )

        mock_client.chat.completions.create.side_effect = api_error

        with pytest.raises(PerplexityAPIError, match="API error: Internal server error"):
            client._make_api_call(sample_pharmacy_data, model="test-model")

    def test_make_api_call_with_custom_model(self, client, sample_pharmacy_data):
        """Test API call uses the specified model."""
        mock_client = client._mock_client
        custom_model = "custom-llama-model"

        # The client fixture sets a default return value, we need to reset the mock
        # to ensure we are asserting the call from this test only.
        mock_client.chat.completions.create.reset_mock()

        # We need to mock the return value for this specific call since reset_mock clears it
        success_content = '{"classification": "independent", "confidence": 0.9, "is_compounding": true, "explanation": "Test explanation"}'
        mock_response = create_mock_response(success_content)
        mock_client.chat.completions.create.return_value = mock_response

        client._make_api_call(sample_pharmacy_data, model=custom_model)

        mock_client.chat.completions.create.assert_called_once()
        kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == custom_model
