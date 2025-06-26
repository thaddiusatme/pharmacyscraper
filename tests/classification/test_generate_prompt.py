"""
Tests for the _generate_prompt method in PerplexityClient.
"""
import pytest
from unittest.mock import MagicMock
from pharmacy_scraper.classification.perplexity_client import PerplexityClient

class TestGeneratePrompt:
    """Test cases for the _generate_prompt method."""
    
    @pytest.fixture
    def client(self):
        """Create a test client with a mock API key."""
        return PerplexityClient(api_key="test_key")
    
    def test_generate_prompt_basic(self, client):
        """Test prompt generation with minimal pharmacy data."""
        pharmacy_data = {
            "name": "Test Pharmacy",
            "address": "123 Test St, Testville, TS 12345",
            "categoryName": "Pharmacy, Health",
            "website": "testpharmacy.com"
        }
        
        prompt = client._generate_prompt(pharmacy_data)
        
        # Basic assertions about the prompt structure
        assert "You are an expert pharmacy classifier" in prompt
        assert "RULES:" in prompt
        assert "RESPONSE FORMAT:" in prompt
        assert "EXAMPLES:" in prompt
        
        # Check that the pharmacy data is included
        assert "Name: Test Pharmacy" in prompt
        assert "Address: 123 Test St, Testville, TS 12345" in prompt
        assert "Categories: Pharmacy, Health" in prompt
        assert "Website: testpharmacy.com" in prompt
        
        # Check for required JSON structure in the prompt
        assert '"classification": "independent|chain|hospital|not_a_pharmacy"' in prompt
        assert '"is_compounding": true|false' in prompt
        assert '"confidence": 0.0-1.0' in prompt
        assert '"explanation": "Your explanation here..."' in prompt
    
    def test_generate_prompt_missing_fields(self, client):
        """Test prompt generation with missing optional fields."""
        pharmacy_data = {
            "name": "Test Pharmacy"
            # Missing address, categories, website
        }
        
        prompt = client._generate_prompt(pharmacy_data)
        
        # Should still include the name
        assert "Name: Test Pharmacy" in prompt
        
        # Missing fields should show as 'N/A'
        assert "Address: N/A" in prompt
        assert "Categories: N/A" in prompt
        assert "Website: N/A" in prompt
    
    def test_generate_prompt_with_title_instead_of_name(self, client):
        """Test that the prompt uses 'title' field if 'name' is not present."""
        pharmacy_data = {
            "title": "Test Pharmacy",  # Using title instead of name
            "address": "123 Test St"
        }
        
        prompt = client._generate_prompt(pharmacy_data)
        
        # Should use title when name is not present
        assert "Name: Test Pharmacy" in prompt
    
    def test_generate_prompt_examples_included(self, client):
        """Test that the prompt includes the expected examples."""
        pharmacy_data = {"name": "Test"}
        prompt = client._generate_prompt(pharmacy_data)
        
        # Check that all examples are included
        assert "Main Street Pharmacy" in prompt
        assert "CVS Pharmacy" in prompt
        assert "Union Avenue Compounding Pharmacy" in prompt
        assert "ANMC Pharmacy" in prompt
        
        # Check that example outputs are included
        assert '"classification": "independent"' in prompt
        assert '"classification": "chain"' in prompt
        assert '"classification": "hospital"' in prompt
        assert '"is_compounding": true' in prompt
        assert '"is_compounding": false' in prompt
    
    def test_generate_prompt_rules_section(self, client):
        """Test that the rules section is properly formatted."""
        pharmacy_data = {"name": "Test"}
        prompt = client._generate_prompt(pharmacy_data)
        
        # Check for key rules
        assert "RULES:" in prompt
        rules_section = prompt.split("RULES:")[1].split("RESPONSE FORMAT:")[0]
        
        # Check that all rule categories are present
        assert "1. An 'independent' pharmacy is" in rules_section
        assert "2. A 'chain' pharmacy is" in rules_section
        assert "3. A 'hospital' pharmacy is" in rules_section
        assert "4. 'not_a_pharmacy'" in rules_section
        assert "5. A 'compounding' pharmacy" in rules_section
    
    def test_generate_prompt_response_format(self, client):
        """Test that the response format section is correct."""
        pharmacy_data = {"name": "Test"}
        prompt = client._generate_prompt(pharmacy_data)
        
        # Check response format section exists
        assert "RESPONSE FORMAT:" in prompt
        response_section = prompt.split("RESPONSE FORMAT:")[1].split("EXAMPLES:")[0]
        
        # Check that the response format instructions are present
        assert "Respond with a single JSON object containing" in response_section
        assert "1. classification: 'independent', 'chain', 'hospital', or 'not_a_pharmacy'" in response_section
        assert "2. is_compounding: true or false" in response_section
        assert "3. confidence: 0.0 to 1.0 (1.0 = highest confidence)" in response_section
        assert "4. explanation: A brief explanation of your reasoning (1-3 sentences)" in response_section
        
        # Check that the example JSON format is included
        assert "```json" in prompt
        assert "{\n  \"classification\": \"independent|chain|hospital|not_a_pharmacy\",\n  \"is_compounding\": true|false,\n  \"confidence\": 0.0-1.0,\n  \"explanation\": \"Your explanation here...\"\n}" in prompt
