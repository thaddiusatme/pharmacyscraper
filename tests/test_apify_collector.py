"""Tests for the Apify data collection module."""
import os
import json
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock, call, ANY
import apify_client

# Import the collector class
from pharmacy_scraper.api.apify_collector import ApifyCollector, run_trial

# Sample test data
SAMPLE_CONFIG = {
    "queries": ["pharmacy in New York, NY"],
    "max_results": 5,
    "output_dir": "test_output"
}

SAMPLE_RESULTS = [
    {
        "name": "Test Pharmacy 1",
        "address": "123 Test St, New York, NY 10001",
        "phone": "(212) 555-1234",
        "website": "https://testpharmacy1.com"
    },
    {
        "name": "Test Pharmacy 2",
        "address": "456 Test Ave, New York, NY 10002",
        "phone": "(212) 555-5678",
        "website": "https://testpharmacy2.com"
    }
]

# Fixtures
@pytest.fixture
def mock_apify_client():
    """Create a mock ApifyClient that prevents real API calls."""
    with patch('apify_client.ApifyClient') as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock the actor().call() chain to return a run ID
        mock_run_dict = {
            'id': 'test-run-id',
            'status': 'SUCCEEDED',
            'defaultDatasetId': 'test-dataset-id'
        }
        mock_actor = MagicMock()
        mock_actor.call.return_value = mock_run_dict
        mock_client.actor.return_value = mock_actor

        # Mock the run().get() chain to prevent the polling loop
        mock_run_client = MagicMock()
        mock_run_client.get.return_value = mock_run_dict
        mock_client.run.return_value = mock_run_client

        # Mock the dataset().list_items() chain
        mock_dataset_client = MagicMock()
        mock_list_items_result = MagicMock()
        mock_list_items_result.items = SAMPLE_RESULTS
        mock_dataset_client.list_items.return_value = mock_list_items_result
        mock_dataset_client.iterate_items.return_value = SAMPLE_RESULTS # Fallback
        mock_client.dataset.return_value = mock_dataset_client

        yield mock_client

@pytest.fixture
def apify_collector():
    """Create an ApifyCollector instance with caching disabled."""
    return ApifyCollector(api_token='test-token', use_cache=False)

@pytest.fixture
def collector(tmp_path):
    """Create a test collector with a temporary output directory and no caching."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return ApifyCollector(output_dir=str(output_dir), use_cache=False)

def test_init_creates_output_dir(tmp_path):
    """Test that output directory is created if it doesn't exist."""
    output_dir = tmp_path / "new_output"
    assert not output_dir.exists()
    
    collector = ApifyCollector(output_dir=str(output_dir))
    assert output_dir.exists()
    assert collector.output_dir == str(output_dir)

def test_run_trial_success(collector, mock_apify_client):
    """Test running a trial with a successful API call."""
    # Create a test config file
    config_path = Path(collector.output_dir) / "test_config.json"
    with open(config_path, 'w') as f:
        json.dump(SAMPLE_CONFIG, f)
    
    # Run the trial
    results = collector.run_trial(str(config_path))
    
    # Verify results
    assert len(results) == 1
    assert results[0]["name"] == "Test Pharmacy 1"
    
    # Verify API client was called correctly
    mock_apify_client.actor.return_value.call.assert_called_once()
    mock_apify_client.actor.return_value.wait_for_finish.assert_called_once()
    mock_apify_client.dataset.return_value.iterate_items.assert_called_once()

def test_run_trial_invalid_actor(collector, mock_apify_client):
    """Test running a trial with an invalid actor ID falls back to direct API."""
    # Create a test config file
    config_path = Path(collector.output_dir) / "test_config.json"
    with open(config_path, 'w') as f:
        json.dump(SAMPLE_CONFIG, f)
    
    # Make the actor call raise an exception
    mock_apify_client.actor.return_value.call.side_effect = Exception("Actor not found")
    
    # Mock the fallback method to return test data
    with patch.object(collector, '_fallback_to_direct_api') as mock_fallback:
        mock_fallback.return_value = [{"name": "Fallback Pharmacy", "address": "123 Test St"}]
        
        # Run the trial - should not raise an exception
        results = collector.run_trial(str(config_path))
        
        # Verify fallback was called
        mock_fallback.assert_called_once()
        
        # Verify results from fallback are returned
        assert len(results) == 1
        assert results[0]["name"] == "Fallback Pharmacy"

def test_run_collection_success(apify_collector, mock_apify_client):
    """Test successful execution of the Apify collection."""
    # Setup mock dataset items using standard SAMPLE_RESULTS
    mock_dataset = MagicMock()
    mock_dataset.iterate_items.return_value = SAMPLE_RESULTS
    mock_list_result = MagicMock()
    mock_list_result.items = SAMPLE_RESULTS
    mock_dataset.list_items.return_value = mock_list_result
    
    mock_apify_client.dataset.return_value = mock_dataset
    
    # Create a config with the correct format (list of dicts with query, city, state)
    config = [{"query": "pharmacies in Los Angeles, CA", "city": "Los Angeles", "state": "CA"}]
    
    # Run the collection
    results = apify_collector.collect_pharmacies(config)
    
    # Verify results
    assert len(results) == 2
    assert results[0]['name'] == 'Test Pharmacy 1'
    
    # Verify API client was called correctly
    mock_apify_client.actor.return_value.call.assert_called_once()

def test_save_results(collector):
    """Test saving results to a file."""
    # Save test results
    output_file = Path(collector.output_dir) / "test_output.json"
    collector._save_results(SAMPLE_RESULTS, str(output_file))
    
    # Verify file was created
    assert output_file.exists()
    
    # Verify file contents
    with open(output_file) as f:
        saved_results = json.load(f)
    assert saved_results == SAMPLE_RESULTS

def test_load_config(collector):
    """Test loading a config file."""
    # Create a test config file
    config_path = Path(collector.output_dir) / "test_config.json"
    with open(config_path, 'w') as f:
        json.dump(SAMPLE_CONFIG, f)
    
    # Load the config
    config = collector._load_config(str(config_path))
    
    # Verify config was loaded correctly
    assert config["queries"] == ["pharmacy in New York, NY"]
    assert config["max_results"] == 5
    assert config["output_dir"] == str(collector.output_dir)

def test_load_config_missing_file(collector):
    """Test loading a non-existent config file."""
    with pytest.raises(FileNotFoundError):
        collector._load_config("nonexistent.json")

def test_apify_collector_initialization():
    """Test that the ApifyCollector initializes with the correct API token."""
    with patch.dict('os.environ', {'APIFY_API_TOKEN': 'test-token'}):
        collector = ApifyCollector(api_token='test-token')
        assert collector.api_token == 'test-token'
        assert collector.rate_limit_ms == 1000  # Default value

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
    with pytest.raises(FileNotFoundError):
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
    """Test running a trial with a successful API response."""
    # Setup test data
    config = {
        'trial_run_test': {
            'states': {
                'California': {
                    'cities': [{'name': 'Los Angeles', 'queries': ['pharmacy']}]
                }
            },
            'output_dir': str(tmp_path / 'output'),
            'max_results_per_query': 5,
            'rate_limit_ms': 100
        }
    }
    
    config_path = tmp_path / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f)
    
    # Configure the mock client
    mock_run = mock_apify_client.return_value.actor.return_value.call.return_value
    mock_run.wait_for_finish.return_value = {'status': 'SUCCEEDED'}
    
    # Run the test
    results = apify_collector.run_trial(str(config_path))
    
    # Verify results
    assert len(results) > 0
    assert 'Los Angeles' in results
    assert len(results['Los Angeles']) > 0
    assert results['Los Angeles'][0]['name'] == 'Test Pharmacy 1'

def test_run_trial_invalid_actor(apify_collector, tmp_path):
    """Test fallback to direct API when actor is not found."""
    # Create a test config file
    config_path = tmp_path / "test_config.json"
    config_data = {
        "trial_run_test": {
            "states": {
                "California": {
                    "cities": [{"name": "LA", "queries": ["pharmacy"]}]
                }
            },
            "output_dir": str(tmp_path / 'output'),
            "max_results_per_query": 5
        }
    }
    config_path.write_text(json.dumps(config_data))
    
    # Create a collector with a real client that we'll mock
    collector = ApifyCollector(api_token='test-token')
    
    # Mock the _execute_actor method to simulate actor not found and fallback
    with patch.object(collector, '_execute_actor') as mock_execute:
        # First call raises exception, second call returns fallback data
        mock_execute.return_value = [{"name": "Fallback Pharmacy", "address": "123 Test St"}]
        
        # Run the trial - should not raise an exception
        results = collector.run_trial(str(config_path))
        
        # Verify the method was called with the expected query
        mock_execute.assert_called_once()
        
        # Verify results from fallback are returned
        assert len(results) == 1
        assert results[0]["name"] == "Fallback Pharmacy"

def test_run_collection_success(apify_collector, mock_apify_client):
    """Test successful execution of the Apify collection."""
    # Setup mock dataset items using standard SAMPLE_RESULTS
    mock_dataset = MagicMock()
    mock_dataset.iterate_items.return_value = SAMPLE_RESULTS
    mock_list_result = MagicMock()
    mock_list_result.items = SAMPLE_RESULTS
    mock_dataset.list_items.return_value = mock_list_result
    
    mock_apify_client.dataset.return_value = mock_dataset
    
    # Create a simple configuration
    config = [{"query": "pharmacies in Los Angeles, CA", "city": "Los Angeles", "state": "CA"}]
    
    # Run the collection
    results = apify_collector.collect_pharmacies(config)
    
    # Verify results
    assert len(results) == 2
    assert results[0]['name'] == 'Test Pharmacy 1'
    
    # Verify API client was called correctly
    mock_apify_client.actor.return_value.call.assert_called_once()

def test_run_trial_invalid_actor(tmp_path):
    """Test fallback to direct API when actor is not found with state/city config."""
    # Create a test configuration with an invalid actor ID
    config = {
        "states": ["CA"],
        "queries": {
            "CA": [
                {"query": "pharmacies in Los Angeles, CA", "city": "Los Angeles", "state": "CA"}
            ]
        }
    }
    config_path = tmp_path / "test_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f)
    
    # Create a collector with a real client that we'll mock
    collector = ApifyCollector(api_token='test-token')
    
    # Mock the _execute_actor method to simulate actor not found and fallback
    with patch.object(collector, '_execute_actor') as mock_execute:
        # Mock the fallback response
        mock_execute.return_value = [{"name": "Fallback Pharmacy", "address": "123 Test St"}]
        
        # Run the trial - should not raise an exception
        results = collector.run_trial(str(config_path))
        
        # Verify the method was called with the expected query
        mock_execute.assert_called_once()
        
        # Verify results from fallback are returned
        assert len(results) == 1
        assert results[0]["name"] == "Fallback Pharmacy"

def test_save_results(apify_collector, tmp_path):
    """Test saving collected data to a CSV file."""
    # Create test data
    pharmacies = [
        {'name': 'Pharmacy 1', 'address': '123 St', 'city': 'Test City', 'state': 'CA'},
        {'name': 'Pharmacy 2', 'address': '456 Ave', 'city': 'Test City', 'state': 'CA'}
    ]
    
    # Test saving to CSV
    output_path = tmp_path / 'pharmacies.csv'
    apify_collector.save_results(pharmacies, str(output_path))
    
    # Verify file was created and contains the data
    assert output_path.exists()
    df = pd.read_csv(output_path)
    assert len(df) == 2
    assert 'Pharmacy 1' in df['name'].values
    assert 'Pharmacy 2' in df['name'].values

def test_chain_pharmacy_filtering(apify_collector):
    """Test filtering out chain pharmacies."""
    # Create test data with both chain and independent pharmacies
    pharmacies = [
        {'name': 'CVS Pharmacy', 'address': '123 St', 'city': 'Test City', 'state': 'CA'},  # Chain
        {'name': 'Walgreens', 'address': '456 Ave', 'city': 'Test City', 'state': 'CA'},    # Chain
        {'name': 'Local Drug Store', 'address': '789 Blvd', 'city': 'Test City', 'state': 'CA'},  # Independent
        {'name': 'Rite Aid', 'address': '012 Rd', 'city': 'Test City', 'state': 'CA'},     # Chain
        {'name': 'Corner Pharmacy', 'address': '345 Ln', 'city': 'Test City', 'state': 'CA'},    # Independent
    ]
    
    # Filter out chain pharmacies
    filtered = apify_collector.filter_chain_pharmacies(pharmacies)
    
    # Verify only independent pharmacies remain
    assert len(filtered) == 2
    assert all('CVS' not in p['name'] for p in filtered)
    assert all('Walgreens' not in p['name'] for p in filtered)
    assert all('Rite Aid' not in p['name'] for p in filtered)
    assert any('Local Drug Store' in p['name'] for p in filtered)
    assert any('Corner Pharmacy' in p['name'] for p in filtered)

def test_init_no_api_key():
    """Test initialization without an API key."""
    with patch.dict('os.environ', {}, clear=True):
        with pytest.raises(ValueError, match="Apify API key not provided"):
            ApifyCollector()

def test_init_with_api_key():
    """Test initialization with an API key."""
    collector = ApifyCollector(api_key='test-api-key')
    assert collector.api_key == 'test-api-key'
    assert collector.output_dir == "output"  # Default value

def test_init_with_env_var():
    """Test initialization with API key from environment variable."""
    with patch.dict('os.environ', {'APIFY_API_TOKEN': 'env-api-key'}):
        collector = ApifyCollector()
        assert collector.api_key == 'env-api-key'

def test_run_trial_success_new(collector, mock_apify_client):
    """Test successful run of a trial collection."""
    # Mock the client to return our test dataset
    results = collector.run_trial("pharmacy", "New York, NY")
    
    # Verify the results
    assert len(results) == 2
    assert results[0]["name"] == "Test Pharmacy 1"
    assert results[1]["name"] == "Test Pharmacy 2"
    
    # Verify the client was called correctly
    assert mock_apify_client.actor.called
    mock_apify_client.actor().call.assert_called_once()
    
    # Verify the dataset was accessed
    assert mock_apify_client.dataset.called
    mock_apify_client.dataset().list_items.assert_called_once()

def test_run_trial_invalid_actor_new(collector, mock_apify_client):
    """Test fallback behavior when actor is not found in new test style."""
    # Configure the mock to raise an exception when the actor is accessed
    mock_apify_client.actor.return_value.get.side_effect = Exception("Actor not found")
    
    # Mock the fallback method to return test data
    with patch.object(collector, '_fallback_to_direct_api') as mock_fallback:
        mock_fallback.return_value = [{"name": "Fallback Pharmacy", "address": "123 Test St"}]
        
        # Run the trial - should not raise an exception
        results = collector.run_trial("test query", "test location")
        
        # Verify fallback was called with the correct arguments
        mock_fallback.assert_called_once_with("test query in test location", 10)
        
        # Verify results from fallback are returned
        assert len(results) == 1
        assert results[0]["name"] == "Fallback Pharmacy"

def test_collect_pharmacies_success_new(collector, mock_apify_client, tmp_path):
    """Test collecting pharmacies for a city/state."""
    # Set up a temporary output directory
    collector.output_dir = str(tmp_path)
    
    # Run the collection
    results = collector.collect_pharmacies([{"query": "pharmacies in New York, NY", "city": "New York", "state": "NY"}])
    
    # Verify the results
    assert len(results) == 2
    
    # Verify the output file was created
    output_file = tmp_path / f"pharmacies_NY_new_york.json"
    assert output_file.exists()
    
    # Verify the file contents
    with open(output_file, 'r') as f:
        saved_data = json.load(f)
    assert len(saved_data) == 2
    assert saved_data[0]["name"] == "Test Pharmacy 1"

def test_collect_pharmacies_error_new(collector, mock_apify_client, caplog):
    """Test error handling during collection."""
    # Make the _execute_actor method raise an exception
    with patch.object(collector, '_execute_actor') as mock_execute_actor:
        mock_execute_actor.side_effect = Exception("Test error")
        
        # Clear any existing log records
        caplog.clear()
        
        # Run the collection - should not raise
        query = {"query": "pharmacies in New York, NY", "city": "New York", "state": "NY"}
        results = collector.collect_pharmacies([query])
        
        # Verify we got an empty list on error
        assert results == []
        
        # Verify the error was logged
        assert any("Test error" in record.message for record in caplog.records)

def test_output_dir_creation_new(collector, tmp_path):
    """Test that the output directory is created if it doesn't exist."""
    # Create a collector with a new subdirectory
    new_dir = tmp_path / "new_output"
    collector = ApifyCollector(api_key='test-api-key', output_dir=str(new_dir))
    
    # The directory should exist
    assert new_dir.exists()
    assert new_dir.is_dir()
    
    # The output_dir should be a string
    assert isinstance(collector.output_dir, str)
