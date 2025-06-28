"""Tests for the API cost tracking functionality."""
import pytest
from unittest.mock import patch, MagicMock, call, mock_open
import json
from pathlib import Path

from pharmacy_scraper.utils.api_cost_tracker import APICostTracker
from pharmacy_scraper.utils.api_usage_tracker import APICreditTracker

# Fixtures
@pytest.fixture
def credit_tracker():
    """Create a clean credit tracker for each test."""
    # Mock the file operations to prevent state persistence between tests
    with patch('pathlib.Path.mkdir'), \
         patch('builtins.open', mock_open()) as mock_file, \
         patch('json.load', return_value={"total_used": 0.0, "daily_usage": {}}):
        tracker = APICreditTracker()
        # Reset the usage data to ensure clean state
        tracker.usage_data = {"total_used": 0.0, "daily_usage": {}}
        yield tracker

# Test Cases
def test_google_places_details_cost_calculation(credit_tracker):
    """Test cost calculation for Google Places details API."""
    # Mock response from Google Places API
    mock_response = {
        'result': {
            'name': 'Test Pharmacy',
            'formatted_address': '123 Test St',
            'place_id': 'test123'
        },
        'status': 'OK'
    }
    
    # Record usage and get cost
    cost = APICostTracker.record_usage_from_response(
        provider='google_places',
        service='details',
        response=mock_response,
        operation='get_place_details',
        credit_tracker=credit_tracker
    )
    
    # Should use the fixed cost for details API
    assert cost == 0.02
    assert credit_tracker.get_total_usage() == 0.02

def test_google_places_nearby_search_cost_calculation(credit_tracker):
    """Test cost calculation for Google Places nearby search."""
    # Mock response with 15 results
    mock_response = {
        'results': [{'name': f'Pharmacy {i}'} for i in range(15)],
        'status': 'OK'
    }
    
    # Record usage and get cost
    cost = APICostTracker.record_usage_from_response(
        provider='google_places',
        service='nearby_search',
        response=mock_response,
        operation='search_nearby',
        credit_tracker=credit_tracker
    )
    
    # Calculate expected cost: 15 results * 0.000032 per result
    expected_cost = 15 * 0.000032
    assert abs(cost - expected_cost) < 0.0001
    assert abs(credit_tracker.get_total_usage() - expected_cost) < 0.0001

def test_apify_cost_calculation(credit_tracker):
    """Test cost calculation for Apify API."""
    # Mock response with 10 items
    mock_response = {
        'items': [{'name': f'Pharmacy {i}'} for i in range(10)]
    }
    
    # Record usage and get cost
    cost = APICostTracker.record_usage_from_response(
        provider='apify',
        service='google_maps_scraper',
        response=mock_response,
        operation='scrape_maps',
        credit_tracker=credit_tracker
    )
    
    # Expected cost: 0.10 base + (10 * 0.001 per result) = 0.11
    expected_cost = 0.11
    assert cost == expected_cost
    assert credit_tracker.get_total_usage() == expected_cost

def test_openai_cost_calculation(credit_tracker):
    """Test cost calculation for OpenAI API."""
    # Mock OpenAI response with token counts
    class Usage:
        def __init__(self):
            self.prompt_tokens = 100
            self.completion_tokens = 50
    
    class MockResponse:
        def __init__(self):
            self.usage = Usage()
    
    mock_response = MockResponse()
    
    # Record usage and get cost for GPT-4
    cost = APICostTracker.record_usage_from_response(
        provider='openai',
        service='gpt-4',
        response=mock_response,
        operation='complete',
        credit_tracker=credit_tracker
    )
    
    # Expected cost: (100 * 0.03/1000) + (50 * 0.06/1000) = 0.006
    expected_cost = (100 * 0.03/1000) + (50 * 0.06/1000)
    assert abs(cost - expected_cost) < 0.0001
    assert abs(credit_tracker.get_total_usage() - expected_cost) < 0.0001

def test_unknown_provider_uses_default_cost(credit_tracker):
    """Test that an unknown provider uses the default cost."""
    cost = APICostTracker.record_usage_from_response(
        provider='unknown_provider',
        service='unknown_service',
        response={},
        credit_tracker=credit_tracker
    )
    
    assert cost == 0.01
    assert credit_tracker.get_total_usage() == 0.01

def test_error_handling_uses_default_cost(credit_tracker):
    """Test that errors in cost calculation fall back to default cost."""
    # This will raise an exception in the cost calculation
    with patch.object(APICostTracker, '_calculate_google_places_cost', side_effect=Exception("Test error")):
        cost = APICostTracker.record_usage_from_response(
            provider='google_places',
            service='details',
            response={},
            credit_tracker=credit_tracker
        )
        
        # Should fall back to default cost on error
        assert cost == 0.01

def test_google_places_error_records_fallback(credit_tracker):
    """Test that errors in Google Places cost calculation are handled."""
    # This will raise an exception in the cost calculation
    with patch.object(APICostTracker, '_calculate_google_places_cost', side_effect=Exception("Test error")):
        cost = APICostTracker.record_usage_from_response(
            provider='google_places',
            service='details',
            response={},
            credit_tracker=credit_tracker
        )
        
        # Should record the fallback cost
        assert credit_tracker.get_total_usage() == 0.01

def test_cost_tracking_across_multiple_calls(credit_tracker):
    """Test that costs are tracked correctly across multiple API calls."""
    # First call
    APICostTracker.record_usage_from_response(
        provider='google_places',
        service='details',
        response={'result': {}, 'status': 'OK'},
        credit_tracker=credit_tracker
    )
    
    # Second call
    APICostTracker.record_usage_from_response(
        provider='google_places',
        service='details',
        response={'result': {}, 'status': 'OK'},
        credit_tracker=credit_tracker
    )
    
    # Should have 0.02 * 2 = 0.04 total credits used
    assert credit_tracker.get_total_usage() == 0.04  # 0.02 * 2

def test_custom_cost_structure(credit_tracker):
    """Test that custom cost structures can be used."""
    # Save original cost structure
    original_costs = APICostTracker.COST_STRUCTURES.copy()
    
    try:
        # Set up custom cost structure
        APICostTracker.COST_STRUCTURES = {
            'custom_provider': {
                'custom_service': {
                    'base': 0.05,
                    'per_item': 0.0003
                }
            }
        }
        
        # Test with custom cost structure
        mock_response = {'items': [1, 2, 3, 4, 5]}  # 5 items
        cost = APICostTracker.record_usage_from_response(
            provider='custom_provider',
            service='custom_service',
            response=mock_response,
            credit_tracker=credit_tracker
        )
        
        # Expected cost: 0.05 + (5 * 0.0003) = 0.0515
        expected_cost = 0.05 + (5 * 0.0003)
        assert abs(cost - expected_cost) < 0.0001
        
    finally:
        # Restore original cost structure
        APICostTracker.COST_STRUCTURES = original_costs