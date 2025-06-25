"""Configuration and fixtures for tests."""
import os
import pytest
from unittest.mock import patch, MagicMock

# Add the project root to the Python path for imports
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture
def mock_apify_client():
    """Fixture that mocks the ApifyClient."""
    with patch('scripts.apify_collector.ApifyClient') as mock_client_class:
        # Create mock client instance and methods
        mock_client = MagicMock()
        mock_actor = MagicMock()
        mock_run = MagicMock()
        mock_dataset = MagicMock()
        
        # Configure the mock client to return our mock actor
        mock_client.actor.return_value = mock_actor
        mock_actor.start.return_value = mock_run
        mock_run.wait_for_finish.return_value = {"defaultDatasetId": "test-dataset"}
        
        # Configure dataset to return mock items
        mock_items = [
            {"name": "Test Pharmacy 1", "address": "123 Test St"},
            {"name": "Test Pharmacy 2", "address": "456 Test Ave"}
        ]
        mock_dataset.iterate_items.return_value = mock_items
        mock_client.dataset.return_value = mock_dataset
        
        # Make the mock class return our mock client instance
        mock_client_class.return_value = mock_client
        
        yield mock_client


@pytest.fixture
def apify_collector(mock_apify_client):
    """Fixture that returns an ApifyCollector instance with a test API token."""
    from src.api.apify_collector import ApifyCollector
    collector = ApifyCollector(api_token="test-token")
    # Ensure the collector is using our mock client
    collector.client = mock_apify_client
    return collector


@pytest.fixture
def sample_states_cities():
    """Sample states and cities for testing."""
    return {
        "California": ["Los Angeles", "San Francisco"],
        "Texas": ["Houston", "Dallas"]
    }
