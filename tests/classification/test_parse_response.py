"""
Tests for the _parse_response method in PerplexityClient.
"""
import json
import pytest
from unittest.mock import MagicMock

from pharmacy_scraper.classification.perplexity_client import PerplexityClient, ResponseParseError

class TestParseResponse:
    """Test cases for the _parse_response method."""

    @pytest.fixture
    def client(self):
        """Create a test client with a mock API key."""
        return PerplexityClient(api_key="test_key", rate_limit=0)

    def create_mock_response(self, content: any) -> MagicMock:
        """Create a mock response object with the given content."""
        response = MagicMock()
        if content is None:
            # Simulate a response where the content of the message is None
            message = MagicMock()
            message.content = None
            choice = MagicMock()
            choice.message = message
            response.choices = [choice]
        elif isinstance(content, list):
            # Simulate a response with a specific list of choices (e.g., empty)
            response.choices = content
        else:
            # Simulate a standard response with string content
            message = MagicMock()
            message.content = str(content)
            choice = MagicMock()
            choice.message = message
            response.choices = [choice]
        return response

    def test_parse_valid_json_response(self, client):
        """Test parsing a valid JSON response."""
        response_data = {
            "classification": "independent",
            "is_compounding": True,
            "confidence": 0.9,
            "explanation": "Test explanation"
        }
        response = self.create_mock_response(json.dumps(response_data))
        result = client._parse_response(response)
        assert result == response_data

    def test_parse_response_with_json_code_block(self, client):
        """Test parsing a response with JSON in a ```json code block."""
        response_data = {
            "classification": "chain",
            "is_compounding": False,
            "confidence": 0.8,
            "explanation": "Test explanation"
        }
        json_content = f"""Some text before.\n```json\n{json.dumps(response_data, indent=2)}\n```\nSome text after."""
        response = self.create_mock_response(json_content)
        result = client._parse_response(response)
        assert result == response_data

    def test_parse_response_with_generic_code_block(self, client):
        """Test parsing a response with JSON in a generic ``` code block."""
        response_data = {
            "classification": "hospital",
            "is_compounding": True,
            "confidence": 0.7,
            "explanation": "Test explanation"
        }
        json_content = f"""```\n{json.dumps(response_data, indent=2)}\n```"""
        response = self.create_mock_response(json_content)
        result = client._parse_response(response)
        assert result == response_data

    def test_parse_response_with_invalid_json(self, client):
        """Test parsing a response with invalid JSON."""
        response = self.create_mock_response("not valid json")
        with pytest.raises(ResponseParseError, match="Failed to parse JSON"):
            client._parse_response(response)

    def test_parse_response_with_missing_required_fields(self, client):
        """Test parsing a response with missing required fields."""
        response_data = {"confidence": 0.9}
        response = self.create_mock_response(json.dumps(response_data))
        with pytest.raises(ResponseParseError, match="Response missing required fields"):
            client._parse_response(response)

    def test_parse_response_with_invalid_classification(self, client):
        """Test parsing a response with an invalid classification value."""
        response_data = {
            "classification": "invalid_value",
            "is_compounding": True,
            "confidence": 0.9,
            "explanation": "Test explanation"
        }
        response = self.create_mock_response(json.dumps(response_data))
        with pytest.raises(ResponseParseError, match="Invalid classification value"):
            client._parse_response(response)

    def test_parse_response_with_invalid_confidence_type(self, client):
        """Test parsing a response with a non-numeric confidence."""
        response_data = {
            "classification": "independent",
            "is_compounding": True,
            "confidence": "not_a_float",
            "explanation": "Test explanation"
        }
        response = self.create_mock_response(json.dumps(response_data))
        with pytest.raises(ResponseParseError, match="Invalid confidence value"):
            client._parse_response(response)

    def test_parse_response_with_confidence_out_of_range(self, client):
        """Test parsing a response with confidence outside the [0, 1] range."""
        response_data = {
            "classification": "independent",
            "is_compounding": True,
            "confidence": 1.5,
            "explanation": "Test explanation"
        }
        response = self.create_mock_response(json.dumps(response_data))
        with pytest.raises(ResponseParseError, match="Invalid confidence value"):
            client._parse_response(response)

    def test_parse_response_with_string_booleans(self, client):
        """Test that string booleans ('true'/'false') are correctly parsed."""
        # Test 'true'
        response_data_true = {
            "classification": "independent",
            "is_compounding": "true",
            "confidence": 0.9,
            "explanation": "Test explanation"
        }
        response_true = self.create_mock_response(json.dumps(response_data_true))
        result_true = client._parse_response(response_true)
        assert result_true["is_compounding"] is True

        # Test 'false'
        response_data_false = {
            "classification": "independent",
            "is_compounding": "false",
            "confidence": 0.9,
            "explanation": "Test explanation"
        }
        response_false = self.create_mock_response(json.dumps(response_data_false))
        result_false = client._parse_response(response_false)
        assert result_false["is_compounding"] is False

    def test_parse_response_with_invalid_is_compounding_type(self, client):
        """Test parsing a response with an invalid type for is_compounding."""
        response_data = {
            "classification": "independent",
            "is_compounding": 123,
            "confidence": 0.9,
            "explanation": "Test explanation"
        }
        response = self.create_mock_response(json.dumps(response_data))
        with pytest.raises(ResponseParseError, match="is_compounding must be a boolean"):
            client._parse_response(response)

    def test_parse_response_with_missing_choices(self, client):
        """Test parsing a response with an empty 'choices' list."""
        response = self.create_mock_response([])
        with pytest.raises(ResponseParseError, match="Response from API has no 'choices'"):
            client._parse_response(response)

    def test_parse_response_with_missing_message(self, client):
        """Test parsing a response where a choice is missing the 'message' attribute."""
        choice = MagicMock()
        choice.message = None
        response = MagicMock()
        response.choices = [choice]
        with pytest.raises(ResponseParseError, match="Response choice missing 'message' or 'content'"):
            client._parse_response(response)

    def test_parse_response_with_missing_content(self, client):
        """Test parsing a response where the message is missing the 'content' attribute."""
        response = self.create_mock_response(None)
        with pytest.raises(ResponseParseError, match="Response choice missing 'message' or 'content'"):
            client._parse_response(response)