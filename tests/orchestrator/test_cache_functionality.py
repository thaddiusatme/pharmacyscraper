"""Tests for the cache functionality in the pipeline orchestrator."""
import json
import logging
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from pharmacy_scraper.orchestrator.pipeline_orchestrator import PipelineOrchestrator, PipelineConfig
from pharmacy_scraper.utils.api_usage_tracker import credit_tracker

# Sample data for testing
SAMPLE_PHARMACY = {
    "name": "Test Pharmacy",
    "address": "123 Test St",
    "city": "San Francisco",
    "state": "CA",
    "zip": "94105",
    "phone": "(555) 123-4567",
    "verified": True
}

class TestCacheFunctionality:
    """Tests for the cache functionality in the pipeline orchestrator."""

    def test_cache_hit(self, tmp_path):
        # Reset credit tracker to ensure a clean state for the test
        credit_tracker.reset()
        """Test that the orchestrator uses the cache on subsequent calls."""
        # Setup test directories
        cache_dir = tmp_path / "test_cache"
        output_dir = tmp_path / "test_output"
        cache_dir.mkdir()
        output_dir.mkdir()
        
        # Create a sample config file
        config_path = tmp_path / "test_config.json"
        sample_config = {
            "cache_dir": str(cache_dir),
            "output_dir": str(output_dir),
            "locations": ["San Francisco, CA"],
            "api_keys": {
                "google_places": "test_google_places_key",
                "perplexity": "test_perplexity_key"
            },
            "max_budget": 100.0,
            "api_cost_limits": {
                "apify": 0.5,
                "google_places": 0.3,
                "perplexity": 0.2
            }
        }
    
        with open(config_path, 'w') as f:
            json.dump(sample_config, f)
        
        # The orchestrator's _execute_pharmacy_query method uses the cache_dir from its config
        # so we will place the cache file there.
        api_cache_dir = cache_dir
        
        # Use the same cache key format as ApifyCollector
        cache_key = "test_query_in_san_francisco_ca"
        cache_file_path = api_cache_dir / f"{cache_key}.json"
        
        # The _execute_pharmacy_query method expects the cache file to contain
        # a direct list of pharmacy records, not a structured object with 'items' key
        cache_content = [SAMPLE_PHARMACY]
        
        with open(cache_file_path, 'w') as f:
            json.dump(cache_content, f)
            
        logger.debug(f"Created cache file at {cache_file_path} with content: {cache_content}")
        
        # Verify the file was created and has content
        assert cache_file_path.exists(), f"Cache file was not created at {cache_file_path}"
        with open(cache_file_path, 'r') as f:
            file_content = json.load(f)
            logger.debug(f"Cache file content after creation: {file_content}")
            # Verify the cache content is a list with our sample pharmacy
            assert isinstance(file_content, list), "Cache file should contain a list of pharmacies"
            assert len(file_content) == 1, "Cache file should contain one pharmacy"
            assert file_content[0] == SAMPLE_PHARMACY, "Cache file pharmacy data mismatch"
        
        # Mock the collector to fail if called
        mock_collector = MagicMock()
        mock_collector.run_trial.side_effect = AssertionError("Collector should not be called when cache exists")
        
        # Mock the classifier
        mock_classifier = MagicMock()
        mock_classifier.classify_pharmacies.return_value = [SAMPLE_PHARMACY]
        
        # Mock the verifier
        mock_verifier = MagicMock()
        mock_verifier.verify_pharmacy.return_value = SAMPLE_PHARMACY
        
        with patch('pharmacy_scraper.api.apify_collector.ApifyCollector', return_value=mock_collector), \
             patch('pharmacy_scraper.classification.classifier.Classifier', return_value=mock_classifier), \
             patch('pharmacy_scraper.verification.google_places.verify_pharmacy', return_value=SAMPLE_PHARMACY), \
             patch('pathlib.Path.mkdir'):
            
            # Create the orchestrator with debug logging
            logger.debug("Creating PipelineOrchestrator instance")
            orchestrator = PipelineOrchestrator(config_path)
            
            # Debug: Log the orchestrator's configuration
            logger.debug(f"Orchestrator config: {orchestrator.config.__dict__}")
            logger.debug(f"Cache dir exists: {Path(orchestrator.config.cache_dir).exists()}")
            logger.debug(f"Cache dir contents: {list(Path(orchestrator.config.cache_dir).glob('*'))}")
            
            # Execute the query - should hit the cache
            logger.debug("Executing pharmacy query - should hit cache")
            
            # Debug: Verify cache file exists before executing query
            logger.debug(f"Cache file exists before query: {cache_file_path.exists()}")
            if cache_file_path.exists():
                with open(cache_file_path, 'r') as f:
                    logger.debug(f"Cache file content before query: {json.load(f)}")
            
            # Execute the query
            result = orchestrator._execute_pharmacy_query("test query", "San Francisco, CA")
            logger.debug(f"Query result: {result}")
            
            # Debug: Check if the cache file was modified
            if cache_file_path.exists():
                with open(cache_file_path, 'r') as f:
                    logger.debug(f"Cache file content after query: {json.load(f)}")
            else:
                logger.debug("Cache file was deleted after query")
            
            # Verify the result - the orchestrator should extract the 'items' from the cache
            # The cache contains {'items': [SAMPLE_PHARMACY], 'cached_at': ...}
            # So we expect the result to be [SAMPLE_PHARMACY]
            assert len(result) == 1
            assert result[0] == SAMPLE_PHARMACY
            
            # Verify the collector was not called
            mock_collector.run_trial.assert_not_called()
            
            # Verify the cache file was used
            assert cache_file_path.exists()
            with open(cache_file_path, 'r') as f:
                cached_data = json.load(f)
