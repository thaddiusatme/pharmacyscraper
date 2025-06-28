import pytest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace
import pandas as pd
import json
from pathlib import Path

from src.pharmacy_scraper.orchestrator.pipeline_orchestrator import PipelineOrchestrator, PipelineConfig
from src.pharmacy_scraper.utils.api_usage_tracker import credit_tracker

@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary config file for testing."""
    config_data = {
        "api_keys": {
            "apify": "test_api_key",
            "google_places": "test_google_key",
            "perplexity": "test_perplexity_key"
        },
        "locations": [
            {
                "state": "CA",
                "cities": ["Test City"],
                "queries": ["test query"]
            }
        ],
        "max_results_per_query": 5,
        "output_dir": str(tmp_path),
        "cache_dir": str(tmp_path / "cache"),
        "verify_places": True
    }
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config_data))
    return str(config_file)

@pytest.fixture
def orchestrator_fixture(temp_config_file):
    """Fixture for PipelineOrchestrator with mocked dependencies."""
    with patch('src.pharmacy_scraper.orchestrator.pipeline_orchestrator.ApifyCollector') as mock_apify_collector, \
         patch('src.pharmacy_scraper.orchestrator.pipeline_orchestrator.Classifier') as mock_classifier, \
         patch('src.pharmacy_scraper.orchestrator.pipeline_orchestrator.remove_duplicates') as mock_remove_duplicates, \
         patch('src.pharmacy_scraper.orchestrator.pipeline_orchestrator.verify_pharmacy') as mock_verify_pharmacy, \
         patch('src.pharmacy_scraper.orchestrator.pipeline_orchestrator.load_from_cache') as mock_load_from_cache, \
         patch('src.pharmacy_scraper.orchestrator.pipeline_orchestrator.save_to_cache') as mock_save_to_cache:

        mock_remove_duplicates.side_effect = lambda df, **kwargs: df
        mock_load_from_cache.return_value = None
        
        orchestrator = PipelineOrchestrator(temp_config_file)
        
        # Group mocks for easy access
        mocks = SimpleNamespace(
            apify=orchestrator.collector,
            classifier=orchestrator.classifier,
            remove_duplicates=mock_remove_duplicates,
            verify_pharmacy=mock_verify_pharmacy,
            cache_load=mock_load_from_cache,
            cache_save=mock_save_to_cache
        )
        
        yield SimpleNamespace(
            orchestrator=orchestrator,
            mocks=mocks
        )

@pytest.fixture(autouse=True)
def reset_credit_tracker():
    """Reset the credit tracker singleton before each test to ensure isolation."""
    credit_tracker.reset()