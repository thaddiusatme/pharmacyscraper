"""
Tests to improve coverage for models.py module.

This test suite focuses on:
1. PharmacyData creation, conversion, and edge cases
2. ClassificationResult properties and methods
3. Enum functionality
4. Edge cases for conversion methods
"""
import pytest
from copy import deepcopy

from pharmacy_scraper.classification.models import (
    PharmacyData,
    ClassificationResult,
    ClassificationSource,
    ClassificationMethod,
    CHAIN_PHARMACY,
    COMPOUNDING_PHARMACY,
    DEFAULT_INDEPENDENT
)


class TestPharmacyData:
    """Tests for PharmacyData class functionality."""
    
    def test_from_dict_all_fields(self):
        """Test creating PharmacyData with all fields populated."""
        data = {
            'name': 'Test Pharmacy',
            'address': '123 Main St',
            'phone': '555-1234',
            'categories': 'Pharmacy, Healthcare',
            'website': 'https://testpharmacy.com',
            'extra_field': 'extra value',
            'another_field': 123
        }
        
        pharmacy = PharmacyData.from_dict(data)
        
        assert pharmacy.name == 'Test Pharmacy'
        assert pharmacy.address == '123 Main St'
        assert pharmacy.phone == '555-1234'
        assert pharmacy.categories == 'Pharmacy, Healthcare'
        assert pharmacy.website == 'https://testpharmacy.com'
        assert pharmacy.raw_data == {'extra_field': 'extra value', 'another_field': 123}
    
    def test_from_dict_minimal(self):
        """Test creating PharmacyData with minimal fields."""
        data = {'name': 'Minimal Pharmacy'}
        
        pharmacy = PharmacyData.from_dict(data)
        
        assert pharmacy.name == 'Minimal Pharmacy'
        assert pharmacy.address is None
        assert pharmacy.phone is None
        assert pharmacy.categories is None
        assert pharmacy.website is None
        assert pharmacy.raw_data == {}
    
    def test_from_dict_title_fallback(self):
        """Test that title field is used when name is not available."""
        data = {'title': 'Title Pharmacy'}
        
        pharmacy = PharmacyData.from_dict(data)
        
        assert pharmacy.name == 'Title Pharmacy'
    
    def test_from_dict_categoryName_fallback(self):
        """Test that categoryName field is used when categories is not available."""
        data = {'name': 'Category Pharmacy', 'categoryName': 'Health & Medical'}
        
        pharmacy = PharmacyData.from_dict(data)
        
        assert pharmacy.categories == 'Health & Medical'
    
    def test_to_dict_all_fields(self):
        """Test converting PharmacyData to dictionary with all fields."""
        pharmacy = PharmacyData(
            name='Test Pharmacy',
            address='123 Main St',
            phone='555-1234',
            categories='Pharmacy, Healthcare',
            website='https://testpharmacy.com',
            raw_data={'extra': 'value'}
        )
        
        result = pharmacy.to_dict()
        
        assert result['name'] == 'Test Pharmacy'
        assert result['address'] == '123 Main St'
        assert result['phone'] == '555-1234'
        assert result['categories'] == 'Pharmacy, Healthcare'
        assert result['website'] == 'https://testpharmacy.com'
        assert result['extra'] == 'value'
    
    def test_to_dict_none_values_excluded(self):
        """Test that None values are excluded from the dictionary."""
        pharmacy = PharmacyData(
            name='Test Pharmacy',
            address=None,
            phone='555-1234',
            categories=None,
            website=None
        )
        
        result = pharmacy.to_dict()
        
        assert 'name' in result
        assert 'phone' in result
        assert 'address' not in result
        assert 'categories' not in result
        assert 'website' not in result
    
    def test_from_dict_empty_dict(self):
        """Test creating PharmacyData from empty dict."""
        empty_dict = {}
        
        pharmacy = PharmacyData.from_dict(empty_dict)
        
        assert pharmacy.name == ''
        assert pharmacy.raw_data == {}


class TestClassificationResult:
    """Tests for ClassificationResult class functionality."""
    
    def test_cached_property_true(self):
        """Test cached property returns True when source is CACHE."""
        result = ClassificationResult(source=ClassificationSource.CACHE)
        assert result.cached is True
    
    def test_cached_property_false(self):
        """Test cached property returns False for other sources."""
        result1 = ClassificationResult(source=ClassificationSource.RULE_BASED)
        result2 = ClassificationResult(source=ClassificationSource.PERPLEXITY)
        result3 = ClassificationResult(source=None)
        
        assert result1.cached is False
        assert result2.cached is False
        assert result3.cached is False
    
    def test_to_dict_all_fields(self):
        """Test converting ClassificationResult with all fields to dictionary."""
        pharmacy_data = PharmacyData(name='Test Pharmacy', address='123 Main St')
        
        result = ClassificationResult(
            classification='independent',
            is_chain=False,
            is_compounding=True,
            confidence=0.95,
            explanation='Test explanation',
            source=ClassificationSource.PERPLEXITY,
            model='pplx-7b-online',
            pharmacy_data=pharmacy_data,
            error=None
        )
        
        dict_result = result.to_dict()
        
        assert dict_result['classification'] == 'independent'
        assert dict_result['is_chain'] is False
        assert dict_result['is_compounding'] is True
        assert dict_result['confidence'] == 0.95
        assert dict_result['explanation'] == 'Test explanation'
        assert dict_result['source'] == 'perplexity'
        assert dict_result['model'] == 'pplx-7b-online'
        assert dict_result['cached'] is False
        assert 'pharmacy_data' in dict_result
        assert dict_result['pharmacy_data']['name'] == 'Test Pharmacy'
    
    def test_to_dict_none_values_excluded(self):
        """Test that None values are excluded from the dictionary."""
        result = ClassificationResult(
            is_chain=False,
            is_compounding=True,
            confidence=0.8
        )
        
        dict_result = result.to_dict()
        
        assert 'is_chain' in dict_result
        assert 'is_compounding' in dict_result
        assert 'confidence' in dict_result
        assert 'classification' not in dict_result
        assert 'explanation' not in dict_result
        assert 'source' not in dict_result
        assert 'model' not in dict_result
        assert 'pharmacy_data' not in dict_result
        assert 'error' not in dict_result
    
    def test_to_dict_with_error(self):
        """Test converting a ClassificationResult with error to dictionary."""
        result = ClassificationResult(
            error="API error occurred",
            source=ClassificationSource.PERPLEXITY
        )
        
        dict_result = result.to_dict()
        
        assert dict_result['error'] == "API error occurred"
        assert dict_result['source'] == "perplexity"


class TestEnums:
    """Tests for the enum classes."""
    
    def test_classification_method_values(self):
        """Test ClassificationMethod enum values."""
        assert ClassificationMethod.RULE_BASED == "rule_based"
        assert ClassificationMethod.LLM == "llm"
        assert ClassificationMethod.CACHED == "cached"
    
    def test_classification_source_values(self):
        """Test ClassificationSource enum values."""
        assert ClassificationSource.RULE_BASED == "rule-based"
        assert ClassificationSource.PERPLEXITY == "perplexity"
        assert ClassificationSource.CACHE == "cache"


class TestCommonResultConstants:
    """Tests for the common result constant objects."""
    
    def test_chain_pharmacy_constant(self):
        """Test the CHAIN_PHARMACY constant."""
        assert CHAIN_PHARMACY.classification == "chain"
        assert CHAIN_PHARMACY.is_chain is True
        assert CHAIN_PHARMACY.is_compounding is False
        assert CHAIN_PHARMACY.confidence == 1.0
        assert CHAIN_PHARMACY.source == ClassificationSource.RULE_BASED
        assert CHAIN_PHARMACY.model is None
    
    def test_compounding_pharmacy_constant(self):
        """Test the COMPOUNDING_PHARMACY constant."""
        assert COMPOUNDING_PHARMACY.classification == "independent"
        assert COMPOUNDING_PHARMACY.is_chain is False
        assert COMPOUNDING_PHARMACY.is_compounding is True
        assert COMPOUNDING_PHARMACY.confidence == 0.95
        assert COMPOUNDING_PHARMACY.source == ClassificationSource.RULE_BASED
    
    def test_default_independent_constant(self):
        """Test the DEFAULT_INDEPENDENT constant."""
        assert DEFAULT_INDEPENDENT.classification == "independent"
        assert DEFAULT_INDEPENDENT.is_chain is False
        assert DEFAULT_INDEPENDENT.is_compounding is False
        assert DEFAULT_INDEPENDENT.confidence == 0.5
        assert DEFAULT_INDEPENDENT.explanation == "No chain identifiers found"
        assert DEFAULT_INDEPENDENT.source == ClassificationSource.RULE_BASED


class TestEdgeCases:
    """Tests for edge cases in the models."""
    
    def test_pharmacy_data_constructor_behavior(self):
        """Test the behavior of PharmacyData with dictionaries."""
        # Test 1: When no raw_data is provided, each instance gets its own empty dict
        pharmacy1 = PharmacyData(name='Test1')
        pharmacy2 = PharmacyData(name='Test2')
        
        # Modify the first instance's raw_data
        pharmacy1.raw_data['key'] = 'value'
        
        # Verify the second instance is unaffected
        assert 'key' not in pharmacy2.raw_data
        
        # Test 2: When the same dict is provided to multiple instances, they share the reference
        # This is standard dataclass behavior - it doesn't create copies of mutable objects
        shared_data = {'shared': 'value'}
        pharmacy3 = PharmacyData(name='Test3', raw_data=shared_data)
        pharmacy4 = PharmacyData(name='Test4', raw_data=shared_data)
        
        # When modifying through one instance, the change is visible in both
        # because they reference the same underlying dictionary
        pharmacy3.raw_data['added'] = 'new_value'
        assert 'added' in pharmacy4.raw_data
        
        # Test 3: When the original dict is modified after instance creation
        # the instance's raw_data reflects those changes (shared reference)
        shared_data['external_change'] = 'external'
        assert 'external_change' in pharmacy3.raw_data
    
    def test_classification_result_immutability(self):
        """Test that ClassificationResult behaves as expected with immutability."""
        pharmacy_data = PharmacyData(name='Test')
        result = ClassificationResult(pharmacy_data=pharmacy_data)
        
        # Cannot modify frozen dataclass attributes directly
        with pytest.raises(AttributeError):
            result.is_chain = True
