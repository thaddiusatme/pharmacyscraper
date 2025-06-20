"""Tests for the Apify data collection module."""
import os
import json
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock, call

# Import the collector class
from scripts.apify_collector import ApifyCollector, APIFY_ACTOR_ID


def test_apify_collector_initialization():
    """Test that the ApifyCollector initializes with API token."""
    # Test initialization with token
    collector = ApifyCollector(api_token="test-token")
    assert collector is not None
    assert collector.api_token == "test-token"
    
    # Test initialization with environment variables
    with patch.dict(os.environ, {"APIFY_TOKEN": "env-token"}):
        collector = ApifyCollector()
        assert collector.api_token == "env-token"
    
    with patch.dict(os.environ, {"APIFY_API_TOKEN": "alt-env-token"}):
        collector = ApifyCollector()
        assert collector.api_token == "alt-env-token"
    
    # Test initialization without token
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError):
            ApifyCollector()


def test_load_config(apify_collector, tmp_path):
    """Test loading configuration from a JSON file."""
    # Create a test config file
    config_path = tmp_path / "test_config.json"
    config_data = {
        "trial_run_20240619": {
            "rate_limit_ms": 100,
            "max_results_per_query": 30,
            "output_dir": "data/raw/trial_20240619",
            "states": {
                "California": {
                    "cities": [{"name": "Los Angeles", "queries": ["pharmacy"]}]
                }
            }
        }
    }
    config_path.write_text(json.dumps(config_data))
    
    # Test loading the config
    config = apify_collector._load_config(str(config_path))
    assert config["rate_limit_ms"] == 100
    assert "California" in config["states"]
    assert config["states"]["California"]["cities"][0]["name"] == "Los Angeles"
    
    # Test with non-existent file
    with pytest.raises(Exception):
        apify_collector._load_config("nonexistent.json")


def test_generate_queries_from_config(apify_collector, tmp_path):
    """Test generating queries from configuration."""
    # Create a test config file
    config_path = tmp_path / "test_config.json"
    config_data = {
        "trial_run_20240619": {
            "states": {
                "California": {
                    "cities": [
                        {"name": "Los Angeles", "queries": ["pharmacy 1", "pharmacy 2"]},
                        {"name": "San Francisco", "queries": ["pharmacy 3"]}
                    ]
                },
                "Texas": {
                    "cities": [
                        {"name": "Houston", "queries": ["pharmacy 4"]}
                    ]
                }
            }
        }
    }
    config_path.write_text(json.dumps(config_data))
    
    # Test generating queries
    queries = apify_collector._generate_queries_from_config(str(config_path))
    
    # Should generate 4 queries (2 + 1 + 1)
    assert len(queries) == 4
    
    # Check each query has the expected structure
    for query in queries:
        assert "query" in query
        assert "state" in query
        assert "city" in query
        assert query["state"] in ["California", "Texas"]
        assert query["city"] in ["Los Angeles", "San Francisco", "Houston"]


def test_create_output_directories(apify_collector, tmp_path):
    """Test creating output directories."""
    output_dir = tmp_path / "output"
    state_dirs = ["California", "Texas"]
    
    # Test directory creation
    apify_collector._create_output_directories(str(output_dir), state_dirs)
    
    # Verify directories were created
    assert output_dir.exists()
    for state in state_dirs:
        state_dir = output_dir / state.lower().replace(" ", "_")
        assert state_dir.exists()


def test_run_trial_success(apify_collector, mock_apify_client, tmp_path):
    """Test successful trial run with mocked Apify client."""
    # Create a test config file
    config_path = tmp_path / "test_config.json"
    config_data = {
        "trial_run_20240619": {
            "rate_limit_ms": 100,
            "max_results_per_query": 5,
            "output_dir": str(tmp_path / "output"),
            "states": {
                "California": {
                    "cities": [
                        {"name": "Los Angeles", "queries": ["pharmacy"]}
                    ]
                }
            }
        }
    }
    config_path.write_text(json.dumps(config_data))
    
    # Setup mock Apify client
    mock_run = MagicMock()
    mock_run.wait_for_finish.return_value = None
    mock_run["defaultDatasetId"] = "test-dataset"
    
    mock_dataset = MagicMock()
    mock_dataset.iterate_items.return_value = [
        {"name": "Test Pharmacy", "address": "123 Test St"}
    ]
    
    mock_actor = MagicMock()
    mock_actor.start.return_value = mock_run
    mock_apify_client.actor.return_value = mock_actor
    mock_apify_client.dataset.return_value = mock_dataset
    
    # Run the trial
    results = apify_collector.run_trial(str(config_path))
    
    # Verify results
    assert len(results) == 1
    assert results[0]["name"] == "Test Pharmacy"
    
    # Verify output directory was created
    output_dir = tmp_path / "output" / "california"
    assert output_dir.exists()
    
    # Verify output file was created
    output_file = output_dir / "los_angeles.json"
    assert output_file.exists()
    
    # Verify mock calls
    mock_apify_client.actor.assert_called_once_with(APIFY_ACTOR_ID)
    mock_actor.start.assert_called_once()
    mock_dataset.iterate_items.assert_called_once()


def test_run_trial_invalid_actor(apify_collector, mock_apify_client, tmp_path):
    """Test handling of invalid actor ID."""
    # Create a test config file
    config_path = tmp_path / "test_config.json"
    config_data = {
        "trial_run_20240619": {
            "states": {
                "California": {
                    "cities": [{"name": "LA", "queries": ["pharmacy"]}]
                }
            }
        }
    }
    config_path.write_text(json.dumps(config_data))
    
    # Mock the client to raise an error for invalid actor
    mock_apify_client.actor.side_effect = Exception("Actor not found")
    
    # Run the trial and verify error handling
    with pytest.raises(Exception, match="Apify actor error"):
        apify_collector.run_trial(str(config_path))


def test_generate_search_queries(apify_collector, sample_states_cities):
    """Test generation of search queries for different states and cities."""
    queries = apify_collector.generate_search_queries(sample_states_cities)
    
    # Should generate 4 queries (2 states × 2 cities each)
    assert len(queries) == 4
    
    # Check that each query has the expected structure
    for query in queries:
        assert 'query' in query
        assert 'state' in query
        assert 'city' in query
        assert 'independent pharmacy' in query['query'].lower()
        assert query['state'] in sample_states_cities
        assert query['city'] in sample_states_cities[query['state']]


def test_run_collection_success(apify_collector, mock_apify_client, sample_states_cities):
    """Test successful execution of the Apify collection."""
    # Setup mock Apify client
    mock_run = MagicMock()
    mock_run.wait_for_finish.return_value = None
    mock_run["defaultDatasetId"] = "test-dataset"
    
    mock_dataset = MagicMock()
    mock_dataset.iterate_items.return_value = [
        {"name": "Test Pharmacy", "address": "123 Test St"}
    ]
    
    mock_actor = MagicMock()
    mock_actor.start.return_value = mock_run
    mock_apify_client.actor.return_value = mock_actor
    mock_apify_client.dataset.return_value = mock_dataset
    
    # Generate test queries
    queries = apify_collector.generate_search_queries(sample_states_cities)
    
    # Run the collection
    results = apify_collector.collect_pharmacies(queries)
    
    # Verify the results
    assert len(results) > 0
    assert all('name' in item for item in results)
    assert all('search_state' in item for item in results)
    assert all('search_city' in item for item in results)
    assert all('collected_at' in item for item in results)
    
    # Verify the mock was called as expected
    assert mock_apify_client.actor.call_count == len(queries)
    assert all(call[0][0] == APIFY_ACTOR_ID for call in mock_apify_client.actor.call_args_list)
    assert mock_apify_client.dataset.return_value.iterate_items.call_count == len(queries)


def test_save_results(apify_collector, tmp_path):
    """Test saving collected data to a CSV file."""
    # Create test data
    test_data = [
        {"name": "Pharmacy 1", "address": "123 St", "city": "Test City", "state": "TS"},
        {"name": "Pharmacy 2", "address": "456 Ave", "city": "Test City", "state": "TS"}
    ]
    
    # Test saving to a temporary file
    output_path = tmp_path / "test_output.csv"
    apify_collector.save_results(test_data, str(output_path))
    
    # Verify file was created
    assert output_path.exists()
    
    # Verify content
    df = pd.read_csv(output_path)
    assert len(df) == 2
    assert set(df['name']) == {"Pharmacy 1", "Pharmacy 2"}


def test_chain_pharmacy_filtering(apify_collector):
    """Test filtering out chain pharmacies."""
    test_items = [
        {"name": "CVS Pharmacy", "address": "123 St"},  # Should be filtered
        {"name": "Local Drug Store", "address": "456 Ave"},  # Should be kept
        {"name": "Walgreens", "address": "789 Blvd"},  # Should be filtered
        {"name": "Independent Pharmacy", "address": "101 Main St"}  # Should be kept
    ]
    
    # Test with filtering on
    filtered = [item for item in test_items 
               if apify_collector._is_valid_pharmacy(item, filter_chains=True)]
    assert len(filtered) == 2
    assert all("CVS" not in item["name"] for item in filtered)
    assert all("Walgreens" not in item["name"] for item in filtered)
    
    # Test with filtering off
    not_filtered = [item for item in test_items 
                   if apify_collector._is_valid_pharmacy(item, filter_chains=False)]
    assert len(not_filtered) == 4
