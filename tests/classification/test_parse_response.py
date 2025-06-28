"""
Tests for the _parse_response method in PerplexityClient.
"""
import json
import pytest
from unittest.mock import MagicMock, patch
from typing import Dict, Any

from pharmacy_scraper.classification.perplexity_client import PerplexityClient, PerplexityAPIError

class TestParseResponse:
    """Test cases for the _parse_response method."""
    
    @pytest.fixture
    def client(self):
        """Create a test client with a mock API key."""
        return PerplexityClient(api_key="test_key")
    
    def create_mock_response(self, content: str) -> MagicMock:
        """Create a mock response object with the given content."""
        message = MagicMock()
        message.content = content
        
        choice = MagicMock()
        choice.message = message
        
        response = MagicMock()
        response.choices = [choice]
        return response
    
    def test_parse_valid_json_response(self, client):
        """Test parsing a valid JSON response."""
        # Arrange
        response_data = {
            "classification": "independent",
            "is_compounding": True,
            "confidence": 0.9,
            "explanation": "Test explanation"
        }
        response = self.create_mock_response(json.dumps(response_data))
        
        # Act
        result = client._parse_response(response)
        
        # Assert
        assert result == response_data
    
    def test_parse_response_with_json_code_block(self, client):
        """Test parsing a response with JSON in a code block."""
        # Arrange
        response_data = {
            "classification": "chain",
            "is_compounding": False,
            "confidence": 0.8,
            "explanation": "Test explanation"
        }
        json_content = f"""Here's the classification:
```json
{json.dumps(response_data, indent=2)}
```
This is an explanation."""
        response = self.create_mock_response(json_content)
        
        # Act
        result = client._parse_response(response)
        
        # Assert
        assert result == response_data
    
    def test_parse_response_with_generic_code_block(self, client):
        """Test parsing a response with generic code block."""
        # Arrange
        response_data = {
            "classification": "hospital",
            "is_compounding": True,
            "confidence": 0.7,
            "explanation": "Test explanation"
        }
        json_content = f"""Here's the classification:
```
{json.dumps(response_data, indent=2)}
```
This is an explanation."""
        response = self.create_mock_response(json_content)
        
        # Act
        with patch('pharmacy_scraper.classification.perplexity_client.logger') as mock_logger:
            result = client._parse_response(response)
        
        # Assert - Currently, generic code blocks are not supported, so we expect None
        # This test documents the current behavior, which could be improved in the future
        assert result is None
        mock_logger.error.assert_called()
    
    def test_parse_response_with_missing_required_fields(self, client):
        """Test parsing a response with missing required fields."""
        # Arrange
        response_data = {
            "is_compounding": True,
            "confidence": 0.9
            # Missing 'classification' and 'explanation'
        }
        response = self.create_mock_response(json.dumps(response_data))
        
        # Act
        result = client._parse_response(response)
        
        # Assert
        assert result is None
    
    def test_parse_response_with_invalid_classification(self, client):
        """Test parsing a response with an invalid classification value."""
        # Arrange
        response_data = {
            "classification": "invalid",
            "is_compounding": True,
            "confidence": 0.9,
            "explanation": "Test explanation"
        }
        response = self.create_mock_response(json.dumps(response_data))
        
        # Act
        result = client._parse_response(response)
        
        # Assert
        assert result is None
    
    def test_parse_response_with_invalid_confidence_range(self, client):
        """Test parsing a response with confidence outside [0, 1] range."""
        # Test confidence > 1.0
        response_data_high = {
            "classification": "independent",
            "is_compounding": True,
            "confidence": 1.5,
            "explanation": "Test explanation"
        }
        response_high = self.create_mock_response(json.dumps(response_data_high))
        
        # Act
        result_high = client._parse_response(response_high)
        
        # Assert
        assert result_high is None
        
        # Test confidence < 0
        response_data_low = {
            "classification": "independent",
            "is_compounding": True,
            "confidence": -0.5,
            "explanation": "Test explanation"
        }
        response_low = self.create_mock_response(json.dumps(response_data_low))
        
        # Act
        result_low = client._parse_response(response_low)
        
        # Assert
        assert result_low is None
    
    def test_parse_response_with_invalid_is_compounding(self, client):
        """Test parsing a response with invalid is_compounding value."""
        # Test with string 'true'
        response_data_str = {
            "classification": "independent",
            "is_compounding": "true",
            "confidence": 0.9,
            "explanation": "Test explanation"
        }
        response_str = self.create_mock_response(json.dumps(response_data_str))
        
        # Act
        result = client._parse_response(response_str)
        
        # Assert - string 'true' should be converted to boolean True
        assert result is not None
        assert result["is_compounding"] is True
        
        # Test with invalid type
        response_data_invalid = {
            "classification": "independent",
            "is_compounding": 123,  # Invalid type
            "confidence": 0.9,
            "explanation": "Test explanation"
        }
        response_invalid = self.create_mock_response(json.dumps(response_data_invalid))
        
        # Act
        result_invalid = client._parse_response(response_invalid)
        
        # Assert - should return None for invalid is_compounding type
        assert result_invalid is None
    
    def test_parse_response_with_missing_explanation(self, client):
        """Test parsing a response with missing explanation."""
        # Arrange
        response_data = {
            "classification": "independent",
            "is_compounding": True,
            "confidence": 0.9
            # Missing 'explanation'
        }
        response = self.create_mock_response(json.dumps(response_data))
        
        # Act
        result = client._parse_response(response)
        
        # Assert
        assert result["explanation"] == "No explanation provided by the model"
    
    def test_parse_response_with_explanation_after_code_block(self, client):
        """Test extracting explanation that comes after a code block."""
        # Arrange
        response_data = {
            "classification": "independent",
            "is_compounding": True,
            "confidence": 0.9,
            "explanation": ""  # Empty explanation in JSON
        }
        json_content = f"""```json
{json.dumps(response_data, indent=2)}
```
This is an explanation that comes after the code block."""
        response = self.create_mock_response(json_content)
        
        # Act
        result = client._parse_response(response)
        
        # Assert
        assert result["explanation"] == "This is an explanation that comes after the code block."
    
    def test_parse_response_with_invalid_json(self, client):
        """Test parsing a response with invalid JSON."""
        # Arrange
        response = self.create_mock_response("This is not valid JSON")
        
        # Act
        result = client._parse_response(response)
        
        # Assert
        assert result is None
    
    def test_parse_response_with_missing_choices(self, client):
        """Test parsing a response with missing choices."""
        # Arrange
        response = MagicMock()
        response.choices = []  # Empty choices list
        
        # Act
        result = client._parse_response(response)
        
        # Assert
        assert result is None
    
    def test_parse_response_with_missing_message(self, client):
        """Test parsing a response with missing message."""
        # Arrange
        choice = MagicMock()
        del choice.message  # Remove message attribute
        response = MagicMock()
        response.choices = [choice]
        
        # Act
        result = client._parse_response(response)
        
        # Assert
        assert result is None
    
    def test_parse_response_with_missing_content(self, client):
        """Test parsing a response with missing content."""
        # Arrange
        message = MagicMock()
        del message.content  # Remove content attribute
        choice = MagicMock()
        choice.message = message
        response = MagicMock()
        response.choices = [choice]
        
        # Act
        result = client._parse_response(response)
        
        # Assert
        assert result is None