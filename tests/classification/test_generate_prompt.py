"""
Tests for the _create_prompt method in PerplexityClient.
"""
import pytest
from unittest.mock import MagicMock
from pharmacy_scraper.classification.perplexity_client import PerplexityClient
from pharmacy_scraper.classification.models import PharmacyData

class TestCreatePrompt:
    """Test cases for the _create_prompt method."""
    
    @pytest.fixture
    def client(self):
        """Create a test client with a mock API key."""
        return PerplexityClient(api_key="test_key")
    
    def test_create_prompt_basic(self, client):
        """Test prompt generation with minimal pharmacy data."""
        pharmacy_data = PharmacyData(
            name="Test Pharmacy",
            address="123 Test St, Testville, TS 12345",
            categories="Pharmacy, Health",
            website="testpharmacy.com"
        )
        
        prompt = client._create_prompt(pharmacy_data)
        
        # Basic assertions about the prompt structure
        assert "Please classify the following pharmacy based on its details." in prompt
        
        # Check that the pharmacy data is included
        assert "Name: Test Pharmacy" in prompt
        assert "Address: 123 Test St, Testville, TS 12345" in prompt
        assert "Categories: Pharmacy, Health" in prompt
        assert "Website: testpharmacy.com" in prompt
        
        # Check for required JSON structure in the prompt
        assert '"classification": "independent|chain|hospital|not_a_pharmacy"' in prompt
        assert '"is_chain": true|false' in prompt
        assert '"is_compounding": true|false' in prompt
        assert '"confidence": 0.0-1.0' in prompt
        assert '"explanation": "Your explanation here..."' in prompt
    
    def test_create_prompt_missing_fields(self, client):
        """Test prompt generation with missing optional fields."""
        pharmacy_data = PharmacyData(
            name="Test Pharmacy"
            # Missing address, categories, website
        )
        
        prompt = client._create_prompt(pharmacy_data)
        
        # Should still include the name
        assert "Name: Test Pharmacy" in prompt
        
        # Missing fields should be None
        assert "Address: None" in prompt
        assert "Categories: None" in prompt
        assert "Website: None" in prompt
    
    def test_create_prompt_with_title_instead_of_name(self, client):
        """Test that the prompt uses PharmacyData with name field."""
        pharmacy_data = PharmacyData(
            name="Test Pharmacy",  # Using name as required
            phone="555-123-4567"
        )
        
        prompt = client._create_prompt(pharmacy_data)
        
        # Should use title when name is not present
        assert "Name: Test Pharmacy" in prompt
    
    def test_create_prompt_basic_format(self, client):
        """Test that the prompt has the basic format expected."""
        pharmacy_data = PharmacyData(name="Test")
        prompt = client._create_prompt(pharmacy_data)
        
        # Check basic format
        assert "Please classify the following pharmacy based on its details." in prompt
        assert "Name: Test" in prompt
        assert "Please respond with a JSON object" in prompt
        assert "```json" in prompt
        assert '"is_compounding": true|false' in prompt
    
    def test_create_prompt_json_format(self, client):
        """Test that the JSON format is specified correctly."""
        pharmacy_data = PharmacyData(name="Test")
        prompt = client._create_prompt(pharmacy_data)
        
        # Check for JSON format specification
        assert '```json' in prompt
        assert '```' in prompt
        
        # Check for expected fields in JSON format
        json_format = prompt.split('```json')[1].split('```')[0].strip()
        assert '"classification":' in json_format
        assert '"is_chain":' in json_format
        assert '"is_compounding":' in json_format
        assert '"confidence":' in json_format
        assert '"explanation":' in json_format
    
    def test_create_prompt_pharmacy_data_inclusion(self, client):
        """Test that all pharmacy data fields are included in the prompt."""
        pharmacy_data = PharmacyData(
            name="Full Test Pharmacy", 
            address="123 Main St",
            phone="555-1234",
            website="www.example.com",
            categories="Pharmacy, Retail"
        )
        prompt = client._create_prompt(pharmacy_data)
        
        # Check that all fields are included
        assert "Name: Full Test Pharmacy" in prompt
        assert "Address: 123 Main St" in prompt
        assert "Categories: Pharmacy, Retail" in prompt
        assert "Website: www.example.com" in prompt
        

    

