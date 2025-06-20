"""Tests for the Apify data collection module."""
import os
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

# Import the collector class
from scripts.apify_collector import ApifyCollector


def test_apify_collector_initialization():
    """Test that the ApifyCollector initializes with API token."""
    # Test initialization with token
    collector = ApifyCollector(api_token="test-token")
    assert collector is not None
    assert collector.api_token == "test-token"
    
    # Test initialization with environment variable
    with patch.dict(os.environ, {"APIFY_TOKEN": "env-token"}):
        collector = ApifyCollector()
        assert collector.api_token == "env-token"
    
    # Test initialization without token
    with pytest.raises(ValueError):
        ApifyCollector(api_token=None)


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
    # Generate test queries
    queries = apify_collector.generate_search_queries(sample_states_cities)
    
    # Run the collection
    results = apify_collector.collect_pharmacies(queries)
    
    # Verify the results
    assert len(results) > 0
    assert all('name' in item for item in results)
    assert all('search_state' in item for item in results)
    assert all('search_city' in item for item in results)
    
    # Verify the mock was called as expected
    # Check that actor was called with the correct argument for each query
    assert mock_apify_client.actor.call_count == len(queries)
    assert all(call[0][0] == "apify/google-maps-scraper" for call in mock_apify_client.actor.call_args_list)
    
    # Verify dataset iteration was called for each query
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
