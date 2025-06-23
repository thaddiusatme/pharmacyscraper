"""
Tests for the pharmacy deduplication and self-healing module.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, ANY
from pathlib import Path
from typing import Dict, List, Any
import os
import tempfile

# Import the module to test
from src.dedup_self_heal import (
    group_pharmacies_by_state,
    remove_duplicates,
    identify_underfilled_states,
    self_heal_state,
    merge_new_pharmacies,
    process_pharmacies,
    TARGET_PHARMACIES_PER_STATE,
    DEFAULT_MIN_REQUIRED,
    scrape_pharmacies,
    get_apify_scraper
)
from src.dedup_self_heal.merge_and_dedupe import merge_and_dedupe_files

# Sample test data
@pytest.fixture
def sample_pharmacies():
    """Sample pharmacy data for testing."""
    return pd.DataFrame({
        'name': [
            'Downtown Pharmacy', 
            'Downtown Pharmacy',  # Duplicate
            'Sunset Drugs',
            'Lone Star Pharmacy'
        ],
        'address': [
            '123 Main St',
            '123 Main St',  # Duplicate
            '456 Sunset Blvd',
            '789 Oak St'
        ],
        'city': ['Anytown', 'Anytown', 'Sometown', 'Othertown'],
        'state': ['CA', 'CA', 'CA', 'TX'],
        'zip': ['12345', '12345', '67890', '54321'],
        'phone': ['(555) 123-4567', '(555) 123-4567', '(555) 987-6543', '(555) 456-7890'],
        'is_chain': [False, False, False, False],
        'confidence': [0.95, 0.95, 0.90, 0.85]
    })

def test_group_pharmacies_by_state(sample_pharmacies):
    """Test grouping pharmacies by state."""
    grouped = group_pharmacies_by_state(sample_pharmacies)
    
    assert isinstance(grouped, dict)
    assert set(grouped.keys()) == {'CA', 'TX'}
    assert len(grouped['CA']) == 3  # Includes duplicate
    assert len(grouped['TX']) == 1

def test_remove_duplicates(sample_pharmacies):
    """Test removal of duplicate pharmacies."""
    deduped = remove_duplicates(sample_pharmacies)
    
    assert len(deduped) == 3  # One duplicate removed
    assert 'Downtown Pharmacy' in deduped['name'].values
    assert 'Sunset Drugs' in deduped['name'].values
    assert 'Lone Star Pharmacy' in deduped['name'].values

def test_identify_underfilled_states(sample_pharmacies):
    """Test identification of states needing more pharmacies."""
    grouped = group_pharmacies_by_state(sample_pharmacies)
    underfilled = identify_underfilled_states(grouped, min_required=2)
    
    assert 'TX' in underfilled
    assert underfilled['TX'] == TARGET_PHARMACIES_PER_STATE - 1  # Needs 1 more
    assert 'CA' not in underfilled  # Already has enough

@patch('src.dedup_self_heal.dedup.scrape_pharmacies')
def test_self_heal_state(mock_scrape, sample_pharmacies):
    """Test self-healing for under-filled states."""
    # Mock the scrape_pharmacies function to return test data
    mock_scrape.return_value = pd.DataFrame([{
        'name': 'Empire Drugs',
        'address': '101 Broadway',
        'city': 'New York',
        'state': 'NY',
        'zip': '10001',
        'phone': '(555) 444-4444',
        'is_chain': False,
        'confidence': 0.92
    }])
    
    # Test with empty existing pharmacies
    result = self_heal_state('NY', pd.DataFrame(), 1)
    
    # Should return the new pharmacy
    assert not result.empty
    assert result.iloc[0]['name'] == 'Empire Drugs'
    assert result.iloc[0]['state'] == 'NY'
    
    # Verify scrape_pharmacies was called with the correct arguments
    mock_scrape.assert_called_once()
    args, kwargs = mock_scrape.call_args
    assert args[0] == 'NY'  # state
    assert args[1] > 0      # count
    assert kwargs.get('city') is not None  # Should have a city parameter
    
    # Test with existing pharmacies (should filter out existing)
    existing = pd.DataFrame([{
        'name': 'Existing Pharmacy',
        'address': '123 Test St',
        'city': 'New York',
        'state': 'NY',
        'zip': '10001',
        'phone': '(555) 123-4567',
        'is_chain': False,
        'confidence': 0.95
    }])
    
    # Reset mock for the second test
    mock_scrape.reset_mock()
    mock_scrape.return_value = pd.DataFrame([{
        'name': 'Empire Drugs',  # Same as existing
        'address': '101 Broadway',
        'city': 'New York',
        'state': 'NY',
        'zip': '10001',
        'phone': '(555) 444-4444',
        'is_chain': False,
        'confidence': 0.92
    }])
    
    result = self_heal_state('NY', existing, 1)
    
    # Should return existing + new pharmacies (but new one is a duplicate, so only existing)
    assert len(result) == 1
    assert 'Existing Pharmacy' in result['name'].values
    
    # Verify scrape_pharmacies was called again
    mock_scrape.assert_called_once()

@patch('src.dedup_self_heal.dedup.self_heal_state')
def test_process_pharmacies(mock_self_heal, sample_pharmacies):
    """Test the main processing pipeline for pharmacies."""
    # Create test data with multiple states
    ca_pharmacies = pd.DataFrame([{
        'name': f'CA Pharmacy {i}',
        'address': f'{i} Main St',
        'city': 'Los Angeles',
        'state': 'CA',
        'zip': '90001',
        'phone': f'(555) 111-{i:04d}',
        'is_chain': False,
        'confidence': 0.95
    } for i in range(30)])  # CA has enough pharmacies
    
    ny_pharmacies = pd.DataFrame([{
        'name': f'NY Pharmacy {i}',
        'address': f'{i} Broadway',
        'city': 'New York',
        'state': 'NY',
        'zip': '10001',
        'phone': f'(555) 222-{i:04d}',
        'is_chain': False,
        'confidence': 0.95
    } for i in range(20)])  # NY needs more pharmacies
    
    tx_pharmacies = pd.DataFrame([{
        'name': f'TX Pharmacy {i}',
        'address': f'{i} Main St',
        'city': 'Houston',
        'state': 'TX',
        'zip': '77001',
        'phone': f'(555) 333-{i:04d}',
        'is_chain': False,
        'confidence': 0.95
    } for i in range(10)])  # TX needs more pharmacies
    
    all_pharmacies = pd.concat([ca_pharmacies, ny_pharmacies, tx_pharmacies])
    
    # Mock self_heal_state to return some new pharmacies for NY and TX
    def mock_self_heal_side_effect(state, existing_df, count, min_required):
        if state == 'NY':
            return pd.DataFrame([{
                'name': 'New NY Pharmacy',
                'address': '123 New St',
                'city': 'Buffalo',
                'state': 'NY',
                'zip': '14201',
                'phone': '(555) 444-0001',
                'is_chain': False,
                'confidence': 0.92
            }])
        elif state == 'TX':
            return pd.DataFrame([{
                'name': 'New TX Pharmacy',
                'address': '456 New St',
                'city': 'Austin',
                'state': 'TX',
                'zip': '73301',
                'phone': '(555) 444-0002',
                'is_chain': False,
                'confidence': 0.91
            }])
        return pd.DataFrame()
    
    mock_self_heal.side_effect = mock_self_heal_side_effect
    
    # Process the pharmacies
    result = process_pharmacies(all_pharmacies, min_required=25)
    
    # Verify the result structure
    assert isinstance(result, dict)
    assert set(result.keys()) == {'CA', 'NY', 'TX'}
    
    # CA should have original count (not underfilled)
    assert len(result['CA']) == 30
    
    # NY should have original + 1 new pharmacy
    assert len(result['NY']) == 21
    assert 'New NY Pharmacy' in result['NY']['name'].values
    
    # TX should have original + 1 new pharmacy
    assert len(result['TX']) == 11
    assert 'New TX Pharmacy' in result['TX']['name'].values
    
    # Verify self_heal_state was called for NY and TX but not CA
    assert mock_self_heal.call_count == 2
    called_states = [call[0][0] for call in mock_self_heal.call_args_list]
    assert 'NY' in called_states
    assert 'TX' in called_states
    assert 'CA' not in called_states

def test_merge_new_pharmacies(sample_pharmacies):
    """Test merging new pharmacies with existing ones."""
    existing = sample_pharmacies.head(3)  # First 3 pharmacies
    new = pd.DataFrame([{
        'name': 'Empire Drugs',
        'address': '101 Broadway',
        'city': 'New York',
        'state': 'NY',
        'zip': '10001',
        'phone': '(555) 444-4444',
        'is_chain': False,
        'confidence': 0.92
    }])
    
    # Add a duplicate of the first pharmacy with different confidence
    duplicate = existing.iloc[0].copy()
    duplicate['confidence'] = 0.90
    new = pd.concat([new, duplicate.to_frame().T])
    
    result = merge_new_pharmacies(existing, new)
    
    # Should have original 3 + 1 new - 1 duplicate = 4 total
    assert len(result) == 4
    assert 'Empire Drugs' in result['name'].values
    
    # The duplicate should keep the higher confidence value
    dup_mask = (result['name'] == duplicate['name']) & (result['address'] == duplicate['address'])
    assert result[dup_mask]['confidence'].iloc[0] == 0.95  # Kept the higher confidence

def test_merge_and_dedupe_files():
    """Tests the merge_and_dedupe_files function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy CSV files
        csv1_path = os.path.join(tmpdir, "raw1.csv")
        csv2_path = os.path.join(tmpdir, "raw2.csv")
        output_path = os.path.join(tmpdir, "merged.csv")

        # placeId '1' is a duplicate
        df1 = pd.DataFrame([
            {'placeId': '1', 'name': 'Pharmacy A', 'scrapedAt': '2023-01-01T12:00:00Z'},
            {'placeId': '2', 'name': 'Pharmacy B', 'scrapedAt': '2023-01-01T12:00:00Z'}
        ])
        df1.to_csv(csv1_path, index=False)

        # Keep the newer version of placeId '1'
        df2 = pd.DataFrame([
            {'placeId': '1', 'name': 'Pharmacy A V2', 'scrapedAt': '2023-01-02T12:00:00Z'},
            {'placeId': '3', 'name': 'Pharmacy C', 'scrapedAt': '2023-01-01T12:00:00Z'}
        ])
        df2.to_csv(csv2_path, index=False)

        # Run the function
        merge_and_dedupe_files([os.path.join(tmpdir, "*.csv")], output_path)

        # Check the output
        assert os.path.exists(output_path)
        result_df = pd.read_csv(output_path)

        # Assertions
        assert len(result_df) == 3
        assert 'Pharmacy A V2' in result_df['name'].values
        assert 'Pharmacy A' not in result_df['name'].values # The older one should be dropped
        assert sorted(list(result_df['placeId'])) == ['1', '2', '3']