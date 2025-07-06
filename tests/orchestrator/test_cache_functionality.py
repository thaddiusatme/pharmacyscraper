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
        # Use the exact same cache key format as the orchestrator
        query = "test_query"
        location = "San Francisco, CA"
        cache_key = f"{query}_{location}".lower().replace(" ", "_")
        cache_file_path = cache_dir / f"{cache_key}.json"
        
        logger.debug(f"Using cache key: {cache_key}")
        
        # Create a single test pharmacy record
        cache_content = [SAMPLE_PHARMACY]
        
        # Write it to the cache file
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
        
        # Create a collector mock that will fail if called
        mock_collector = MagicMock()
        mock_collector.run_trial.side_effect = AssertionError("Collector should not be called when cache exists")
        
        # Create the orchestrator
        orchestrator = PipelineOrchestrator(config_path)
        
        # Replace the collector with our mock
        orchestrator.collector = mock_collector
        
        # Debug: Log the orchestrator's configuration
        logger.debug(f"Orchestrator config: {orchestrator.config.__dict__}")
        logger.debug(f"Cache dir exists: {Path(orchestrator.config.cache_dir).exists()}")
        logger.debug(f"Cache dir contents: {list(Path(orchestrator.config.cache_dir).glob('*'))}")
        
        # Debug: Verify cache file exists before executing query
        logger.debug(f"Cache file exists before query: {cache_file_path.exists()}")
        if cache_file_path.exists():
            with open(cache_file_path, 'r') as f:
                logger.debug(f"Cache file content before query: {json.load(f)}")
        
        # Execute the query with the exact same query and location used for the cache key
        result = orchestrator._execute_pharmacy_query(query, location)
        logger.debug(f"Query result: {result}")
        
        # Debug: Check if the cache file was modified
        if cache_file_path.exists():
            with open(cache_file_path, 'r') as f:
                logger.debug(f"Cache file content after query: {json.load(f)}")
        else:
            logger.debug("Cache file was deleted after query")
        
        # Verify the result
        assert len(result) == 1, f"Expected 1 item in result but got {len(result)}"
        assert result[0] == SAMPLE_PHARMACY, "Result content doesn't match expected"
        
        # Verify the collector was not called (should use cache instead)
        mock_collector.run_trial.assert_not_called()
        
        # Verify the cache file still exists
        assert cache_file_path.exists(), "Cache file should still exist after query"
