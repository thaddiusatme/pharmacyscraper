"""Integration tests for the complete pharmacy classification workflow.

These tests verify the end-to-end functionality of the classification system,
including the interaction between rule-based classification and LLM-based classification,
caching behavior, and the full workflow from input data to classification result.
"""

import json
import pytest
import tempfile
from unittest.mock import patch, MagicMock

from pharmacy_scraper.classification.models import (
    PharmacyData, 
    ClassificationResult,
    ClassificationSource
)
from pharmacy_scraper.classification.classifier import Classifier
from pharmacy_scraper.classification.perplexity_client import PerplexityClient


class TestClassificationWorkflow:
    """Test the end-to-end classification workflow."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        # Clear module-level classification cache between tests
        from pharmacy_scraper.classification.classifier import _classification_cache
        _classification_cache.clear()
        
        # Create a temporary directory for cache files
        self.temp_dir = tempfile.TemporaryDirectory()
        
    def teardown_method(self):
        """Clean up after each test."""
        # Remove temporary directory
        self.temp_dir.cleanup()
    
    @patch('pharmacy_scraper.classification.perplexity_client.PerplexityClient.classify_pharmacy')
    def test_rule_based_high_confidence_no_llm_call(self, mock_perplexity_classify):
        """When rule-based classification has high confidence, LLM should not be called."""
        # Setup
        classifier = Classifier()
        
        # Create a pharmacy that should be classified with high confidence by rules
        pharmacy = PharmacyData(
            name="CVS Pharmacy #12345",
            address="123 Main St, Anytown, USA",
            phone=None,
            categories=None,
            website=None
        )
        
        # Perform classification
        result = classifier.classify_pharmacy(pharmacy)
        
        # Verify rule-based classification worked correctly
        assert result.is_chain is True
        assert result.confidence >= 0.9
        assert result.source == ClassificationSource.RULE_BASED
        
        # Verify LLM was not called
        mock_perplexity_classify.assert_not_called()
    
    @patch('pharmacy_scraper.classification.perplexity_client.PerplexityClient.classify_pharmacy')
    def test_rule_based_low_confidence_triggers_llm(self, mock_perplexity_classify):
        """When rule-based classification has low confidence, LLM should be called."""
        # Setup
        # Create an ambiguous pharmacy name with low rule confidence
        pharmacy = PharmacyData(
            name="Family Care Pharmacy",
            address="456 Oak St, Somewhere, USA",
            phone=None,
            categories=None,
            website=None
        )
        
        llm_result = ClassificationResult(
            classification="independent",
            is_chain=False,
            is_compounding=False,
            confidence=0.95,
            explanation="This is an independent pharmacy based on...",
            source=ClassificationSource.PERPLEXITY,
            model="test-model",
            pharmacy_data=pharmacy
        )
        mock_perplexity_classify.return_value = llm_result
        classifier = Classifier()
        
        # Perform classification
        result = classifier.classify_pharmacy(pharmacy)
        
        # Verify LLM was called
        mock_perplexity_classify.assert_called_once()
        
        # Verify we got the LLM result
        assert result.confidence == 0.95
        assert result.source == ClassificationSource.PERPLEXITY
        assert result.is_chain is False
    
    def test_end_to_end_with_perplexity_client(self):
        """Test the full workflow with a mocked API client."""
        # Skip if no API key is available
        pytest.importorskip("openai")
        
        # Create a pharmacy with a name that should have low rule-based confidence
        pharmacy = PharmacyData(
            name="Village Apothecary",
            address="789 Elm St, Elsewhere, USA",
            phone=None,
            categories=None,
            website=None
        )
        
        # Create a test result to return from our patched method
        test_result = ClassificationResult(
            classification="independent",
            is_chain=False,
            is_compounding=True,
            confidence=0.95,
            explanation="This is a mock response for testing",
            source=ClassificationSource.PERPLEXITY,
            model="test-model",
            pharmacy_data=pharmacy
        )
        
        # For this test, we'll patch the classify_pharmacy method directly
        # instead of the lower-level API calls to avoid JSON parsing issues
        with patch('pharmacy_scraper.classification.perplexity_client.PerplexityClient.classify_pharmacy') as mock_classify:
            # Set up the mock to return our test result
            mock_classify.return_value = test_result
            
            # Create a client
            perplexity_client = PerplexityClient(
                api_key="test_key",
                cache_dir=self.temp_dir.name
            )
            
            # Create classifier with our client
            classifier = Classifier(client=perplexity_client)
            
            # First classification should use the LLM
            result1 = classifier.classify_pharmacy(pharmacy)
            
            # Verify the result
            assert result1.is_compounding is True
            assert result1.confidence == 0.95
            assert result1.source == ClassificationSource.PERPLEXITY
            
            # Second classification of the same pharmacy should use cache
            result2 = classifier.classify_pharmacy(pharmacy)
            
            # Verify it came from cache
            assert result2.source == ClassificationSource.CACHE
            
            # Verify the perplexity client's classify_pharmacy was called only once
            assert mock_classify.call_count == 1
    
    def test_batch_classification(self):
        """Test batch classification with mixed rule-based and LLM results."""
        # Create pharmacies for testing - one clear chain, one ambiguous
        pharmacies = [
            PharmacyData(name="Walgreens", address="123 Main St"),
            PharmacyData(name="Community Pharmacy", address="456 Oak St"),
        ]
        
        # Create a test result for the second pharmacy
        test_result = ClassificationResult(
            classification="independent",
            is_chain=False,
            is_compounding=False,
            confidence=0.9,
            explanation="This is a mock response for testing",
            source=ClassificationSource.PERPLEXITY,
            model="test-model",
            pharmacy_data=pharmacies[1]
        )
        
        # Setup patched perplexity client
        with patch('pharmacy_scraper.classification.perplexity_client.PerplexityClient._call_api'):
            # Create a mock perplexity client
            perplexity_client = PerplexityClient(
                api_key="test_key",
                cache_dir=self.temp_dir.name
            )
            
            # Patch the classify_pharmacy method to return our test result
            perplexity_client.classify_pharmacy = MagicMock(return_value=test_result)
            
            # Create classifier with our client
            classifier = Classifier(client=perplexity_client)
            
            # Batch classify
            results = classifier.batch_classify_pharmacies(pharmacies)
            
            # Verify results
            # First should be rule-based (chain)
            assert results[0].is_chain is True
            assert results[0].source == ClassificationSource.RULE_BASED
            
            # Second should be our mocked independent result
            assert results[1].is_chain is False
            assert results[1].source == ClassificationSource.PERPLEXITY
            
            # Verify perplexity client was called only for the second pharmacy
            assert perplexity_client.classify_pharmacy.call_count == 1
    
    @patch('pharmacy_scraper.classification.perplexity_client.PerplexityClient.classify_pharmacy')
    def test_rule_based_compounding_flag_prevents_llm_call(self, mock_classify):
        """When rule-based detects compounding, it should skip LLM even with low confidence."""
        # Setup
        # Create a classifier
        classifier = Classifier()
        
        # Create a pharmacy with 'compounding' in the name
        pharmacy = PharmacyData(
            name="Johnson's Compounding Pharmacy",
            address="123 Health St",
            phone=None,
            categories=None,
            website=None
        )
        
        # Perform classification
        result = classifier.classify_pharmacy(pharmacy)
        
        # Verify compounding was detected
        assert result.is_compounding is True
        
        # Verify source is rule-based and LLM wasn't called
        assert result.source == ClassificationSource.RULE_BASED
        mock_classify.assert_not_called()
    
    @patch('pharmacy_scraper.classification.perplexity_client.PerplexityClient.classify_pharmacy')
    def test_standard_caching_behavior(self, mock_classify):
        """Test that without force_reclassification, second call uses cache."""
        # Create a pharmacy with unique identifiers
        pharmacy = PharmacyData(
            name="Standard Cache Pharmacy", 
            address="123 Cache Test Lane",
            phone="555-123-4567",
            categories=["pharmacy"],
            website=None
        )
        
        # Create a result for perplexity to return
        perplexity_result = ClassificationResult(
            classification="independent",
            is_chain=False,
            is_compounding=False,
            confidence=0.95,
            explanation="Test result for standard caching",
            source=ClassificationSource.PERPLEXITY,
            model="test-model",
            pharmacy_data=pharmacy
        )
        mock_classify.return_value = perplexity_result
        
        # Clear cache
        from pharmacy_scraper.classification.classifier import _classification_cache
        _classification_cache.clear()
        
        # Create client with standard caching
        client = PerplexityClient(
            api_key="test_key",
            cache_dir=self.temp_dir.name,
            cache_enabled=True,
            force_reclassification=False
        )
        
        # Create classifier
        classifier = Classifier(client=client)
        
        # Call classify_pharmacy twice
        result1 = classifier.classify_pharmacy(pharmacy)
        result2 = classifier.classify_pharmacy(pharmacy)
        
        # First call should use API, second should use cache
        assert mock_classify.call_count == 1
        assert result1.source == ClassificationSource.PERPLEXITY
        assert result2.source == ClassificationSource.CACHE
    
    @patch('pharmacy_scraper.classification.perplexity_client.PerplexityClient._call_api_with_retries')
    def test_force_reclassification_bypasses_client_cache(self, mock_call_api):
        """Test that force_reclassification flag bypasses the PerplexityClient's cache."""
        # Create a unique pharmacy for this test
        pharmacy = PharmacyData(
            name="Force Cache Test Pharmacy",
            address="999 Force Ave",
            phone="555-555-1234",
            categories=["pharmacy"],
            website="http://test-pharmacy.com"
        )
        
        # Mock API response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "classification": "independent",
            "is_chain": False,
            "is_compounding": False,
            "confidence": 0.95,
            "explanation": "Test explanation"
        })
        mock_call_api.return_value = mock_response
        
        # Create a fresh temporary directory for this test's cache
        temp_dir = tempfile.TemporaryDirectory()
        
        # CASE 1: Normal caching behavior
        client1 = PerplexityClient(
            api_key="test_key",
            cache_dir=temp_dir.name,
            cache_enabled=True,
            cache_ttl_seconds=3600,  # 1 hour TTL
            force_reclassification=False
        )
        
        # First call should hit the API
        result1 = client1.classify_pharmacy(pharmacy)
        # Second call should use cache
        result2 = client1.classify_pharmacy(pharmacy)
        
        # Verify API call counts and sources
        assert mock_call_api.call_count == 1  # API called once
        assert result1.source == ClassificationSource.PERPLEXITY
        assert result2.source == ClassificationSource.CACHE
        
        # Reset mock for next test
        mock_call_api.reset_mock()
        
        # CASE 2: With force_reclassification=True
        client2 = PerplexityClient(
            api_key="test_key",
            cache_dir=temp_dir.name,  # Using the same cache dir
            cache_enabled=True,
            cache_ttl_seconds=3600,  # 1 hour TTL
            force_reclassification=True  # This is the key setting
        )
        
        # This should bypass the cache and hit the API
        result3 = client2.classify_pharmacy(pharmacy)
        # This should also bypass the cache and hit the API again
        result4 = client2.classify_pharmacy(pharmacy)
        
        # Verify API was called for both requests
        assert mock_call_api.call_count == 2  # API called twice
        assert result3.source == ClassificationSource.PERPLEXITY
        assert result4.source == ClassificationSource.PERPLEXITY
        
        # Clean up
        temp_dir.cleanup()
