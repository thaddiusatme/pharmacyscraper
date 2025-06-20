"""
Tests for the pharmacy deduplication and self-healing module.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, ANY
from pathlib import Path
from typing import Dict, List, Any

# Import the module to test
from src.dedup_self_heal import (
    group_pharmacies_by_state,
    remove_duplicates,
    identify_underfilled_states,
    self_heal_state,
    merge_new_pharmacies,
    process_pharmacies,
    TARGET_PHARMACIES_PER_STATE,
    DEFAULT_MIN_REQUIRED
)

# Sample test data
SAMPLE_PHARMACIES = [
    # California - 3 pharmacies (2 unique)
    {"name": "Downtown Pharmacy", "address": "123 Main St", "city": "San Francisco", "state": "CA", "zip": "94105", "phone": "(555) 111-1111", "is_chain": False, "confidence": 0.95},
    {"name": "Downtown Pharmacy", "address": "123 Main St", "city": "San Francisco", "state": "CA", "zip": "94105", "phone": "(555) 111-1111", "is_chain": False, "confidence": 0.95},  # Duplicate
    {"name": "Sunset Drugs", "address": "456 Sunset Blvd", "city": "Los Angeles", "state": "CA", "zip": "90028", "phone": "(555) 222-2222", "is_chain": False, "confidence": 0.90},
    
    # Texas - 1 pharmacy
    {"name": "Lone Star Pharmacy", "address": "789 Oak St", "city": "Austin", "state": "TX", "zip": "73301", "phone": "(555) 333-3333", "is_chain": False, "confidence": 0.85},
    
    # New York - 1 pharmacy (will be used for testing merge)
    {"name": "Empire Drugs", "address": "101 Broadway", "city": "New York", "state": "NY", "zip": "10001", "phone": "(555) 444-4444", "is_chain": False, "confidence": 0.92}
]

# Sample Apify response data
SAMPLE_APIFY_RESPONSE = [
    {
        'title': 'New Austin Pharmacy',
        'address': '123 New St, Austin, TX 73302',
        'phone': '(555) 555-5555',
        'location': {'lat': 30.2672, 'lng': -97.7431},
        'website': 'https://newaustinpharmacy.com',
        'scrapedAt': '2025-01-01T12:00:00Z'
    },
    {
        'title': 'Austin Compounding',
        'address': '456 Oak Ave, Austin, TX 73303',
        'phone': '(555) 666-6666',
        'location': {'lat': 30.2682, 'lng': -97.7441},
        'website': 'https://austincompounding.com',
        'scrapedAt': '2025-01-01T12:01:00Z'
    }
]

# Fixture for sample data
@pytest.fixture
def sample_pharmacies():
    return pd.DataFrame(SAMPLE_PHARMACIES)

def test_group_pharmacies_by_state(sample_pharmacies):
    """Test grouping pharmacies by state."""
    result = group_pharmacies_by_state(sample_pharmacies)
    
    # Should have 3 states (CA, TX, NY)
    assert len(result) == 3
    assert "CA" in result
    assert "TX" in result
    assert "NY" in result
    
    # Check counts
    assert len(result["CA"]) == 3  # Includes duplicate
    assert len(result["TX"]) == 1
    assert len(result["NY"]) == 1

def test_remove_duplicates(sample_pharmacies):
    """Test removal of duplicate pharmacies."""
    result = remove_duplicates(sample_pharmacies)
    
    # Should have 4 unique pharmacies (original has 5 entries with 1 duplicate)
    assert len(result) == 4
    
    # Check that duplicates are removed (based on name and address)
    names = result["name"].tolist()
    assert names.count("Downtown Pharmacy") == 1

def test_identify_underfilled_states(sample_pharmacies):
    """Test identification of states needing more pharmacies."""
    grouped = group_pharmacies_by_state(sample_pharmacies)
    underfilled = identify_underfilled_states(grouped, min_required=2)
    
    # TX has only 1 pharmacy, needs more (assuming min_required=2)
    assert "TX" in underfilled
    assert "CA" not in underfilled  # CA has 2 unique pharmacies after dedupe
    assert underfilled["TX"] == 1  # Currently has 1, needs 1 more

@patch('src.dedup_self_heal.get_apify_scraper')
def test_scrape_pharmacies(mock_get_scraper, sample_pharmacies):
    """Test scraping pharmacies using Apify."""
    # Mock the Apify scraper
    mock_scraper = MagicMock()
    mock_scraper.scrape_pharmacies.return_value = SAMPLE_APIFY_RESPONSE
    mock_get_scraper.return_value = mock_scraper
    
    # Import here to avoid circular imports
    from src.dedup_self_heal import scrape_pharmacies
    
    # Test with state and count
    result = scrape_pharmacies("TX", 2)
    
    # Check results
    assert len(result) == 2
    assert result.iloc[0]['name'] == 'New Austin Pharmacy'
    assert result.iloc[0]['city'] == ''  # City not provided in the scrape_pharmacies call
    assert 'scraped_at' in result.columns
    
    # Check that the scraper was called with the right arguments
    mock_scraper.scrape_pharmacies.assert_called_once_with(
        state='TX',
        city='',
        query='pharmacy in TX',
        max_results=4  # 2 * 2 (count * 2 for extra results)
    )

@patch('src.dedup_self_heal.scrape_pharmacies')
def test_self_heal_state(mock_scrape, sample_pharmacies):
    """Test self-healing for under-filled states."""
    # Filter to just TX for testing
    tx_pharmacies = sample_pharmacies[sample_pharmacies['state'] == 'TX']
    
    # Mock the scrape_pharmacies function to return new pharmacies
    new_pharmacy = {
        'name': 'New Austin Pharmacy',
        'address': '123 New St',
        'city': 'Austin',
        'state': 'TX',
        'zip': '73302',
        'phone': '(555) 555-5555',
        'is_chain': False,
        'confidence': 0.88,
        'scraped_at': pd.Timestamp.now()
    }
    mock_scrape.return_value = pd.DataFrame([new_pharmacy])
    
    # Test self-healing
    result = self_heal_state("TX", tx_pharmacies, needed=1)
    
    # Should have original + new pharmacy
    assert len(result) == 2
    assert "New Austin Pharmacy" in result["name"].values
    
    # Check that scrape_pharmacies was called
    mock_scrape.assert_called_once()

def test_merge_new_pharmacies(sample_pharmacies):
    """Test merging new pharmacies with existing ones."""
    # Split into existing and new
    existing = sample_pharmacies[sample_pharmacies['state'] != 'NY']
    new = sample_pharmacies[sample_pharmacies['state'] == 'NY']
    
    # Test merge
    result = merge_new_pharmacies(existing, new)
    
    # Should have all pharmacies including the new NY one
    assert len(result) == len(existing) + len(new)
    assert "NY" in result["state"].unique()
    assert "Empire Drugs" in result["name"].values

@patch('src.dedup_self_heal.self_heal_state')
def test_process_pharmacies(mock_self_heal, sample_pharmacies):
    """Test the full processing pipeline."""
    # Mock self_heal_state to just return the input
    mock_self_heal.side_effect = lambda state, df, needed, **kwargs: df
    
    # Process the pharmacies
    result = process_pharmacies(sample_pharmacies, target_per_state=2)
    
    # Should have data for all states
    assert set(result.keys()) == {"CA", "TX", "NY"}
    
    # Check that self_heal_state was called for under-filled states
    # In our test data, only TX has fewer than 2 unique pharmacies
    assert mock_self_heal.call_count == 1
    call_args = mock_self_heal.call_args[1]
    assert call_args["state"] == "TX"
    assert call_args["needed"] == 1  # TX has 1 pharmacy, needs 1 more to reach target of 2