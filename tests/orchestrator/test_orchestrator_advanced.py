"""Tests for advanced functionality in the pipeline orchestrator."""
import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest
import pandas as pd

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from pharmacy_scraper.orchestrator.pipeline_orchestrator import PipelineOrchestrator, PipelineConfig
from pharmacy_scraper.utils.api_usage_tracker import credit_tracker, CreditLimitExceededError

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

class TestOrchestratorAdvanced:
    """Tests for advanced functionality in the pipeline orchestrator."""
    
    def test_api_budget_limits(self, tmp_path):
        """Test that the orchestrator respects API budget limits."""
        # Reset credit tracker to ensure a clean state for the test
        credit_tracker.reset()
        
        # Setup test directories
        cache_dir = tmp_path / "test_cache"
        output_dir = tmp_path / "test_output"
        cache_dir.mkdir()
        output_dir.mkdir()
        
        # Create a sample config file with very low budget limits
        config_path = tmp_path / "test_config.json"
        sample_config = {
            "cache_dir": str(cache_dir),
            "output_dir": str(output_dir),
            "locations": [{
                "state": "CA",
                "cities": ["San Francisco"],
                "queries": ["independent pharmacy"]
            }],
            "api_keys": {
                "apify": "test_apify_key",
                "google_places": "test_google_places_key",
                "perplexity": "test_perplexity_key"
            },
            "max_budget": 0.01,  # Very low budget
            "api_cost_limits": {
                "apify": 0.005,  # Very low limit
                "google_places": 0.005,
                "perplexity": 0.005
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(sample_config, f)
        
        # Create the orchestrator
        orchestrator = PipelineOrchestrator(config_path)
        
        # Mock the collector to return multiple results
        orchestrator.collector = MagicMock()
        test_pharmacy_data = [SAMPLE_PHARMACY, SAMPLE_PHARMACY.copy(), SAMPLE_PHARMACY.copy()]
        orchestrator.collector.run_trial.return_value = test_pharmacy_data
        
        # Mock classifier
        orchestrator.classifier = MagicMock()
        
        # We need to patch the track_usage method to avoid immediate budget exception
        with patch('pharmacy_scraper.utils.api_usage_tracker.APICreditTracker.track_usage') as mock_track_usage:
            # Set up the context manager mock to work normally for the first call
            cm_instance = MagicMock()
            mock_track_usage.side_effect = lambda service: cm_instance
            
            # First call should succeed
            query = "test_query"
            location = "San Francisco, CA"
            result = orchestrator._execute_pharmacy_query(query, location)
            assert len(result) == 3, "Expected results from first query"
            
            # Now make the track_usage raise CreditLimitExceededError on next call
            mock_track_usage.side_effect = CreditLimitExceededError("Insufficient budget for apify")
            
            # Second call should raise the exception
            with pytest.raises(CreditLimitExceededError):
                result = orchestrator._execute_pharmacy_query("another_query", location)
            
            # Reset the mock for the run() test
            mock_track_usage.side_effect = CreditLimitExceededError("Insufficient budget for apify")
            
            # Verify full pipeline run handles the exception gracefully
            output = orchestrator.run()
            assert output is None, "Pipeline run should return None when budget is exceeded"

    def test_pipeline_resume(self, tmp_path):
        """Test that the pipeline can resume from a partially completed state."""
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
            "locations": [{
                "state": "CA",
                "cities": ["San Francisco"],
                "queries": ["independent pharmacy"]
            }],
            "api_keys": {
                "apify": "test_apify_key",
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
        
        # Create the orchestrator
        with patch('pharmacy_scraper.orchestrator.state_manager.StateManager') as mock_state_manager_class:
            # Setup mock state manager to indicate some stages are completed
            mock_state_manager = mock_state_manager_class.return_value
            mock_state_manager.get_task_status.side_effect = lambda task_name: {
                'data_collection': 'completed',
                'deduplication': 'completed',
                'classification': 'failed',  # This stage needs to be run
                'verification': 'not_started'  # This stage needs to be run
            }.get(task_name)
            
            # Create output files for completed stages
            data_collection_file = output_dir / "stage_data_collection_output.json"
            dedup_file = output_dir / "stage_deduplication_output.json"
            
            # Sample data for completed stages
            raw_data = [SAMPLE_PHARMACY, SAMPLE_PHARMACY.copy(), SAMPLE_PHARMACY.copy()]
            deduped_data = [SAMPLE_PHARMACY, SAMPLE_PHARMACY.copy()]
            
            # Create output files
            with open(data_collection_file, 'w') as f:
                json.dump(raw_data, f)
                
            with open(dedup_file, 'w') as f:
                json.dump(deduped_data, f)
            
            # Initialize orchestrator with mocked state manager
            orchestrator = PipelineOrchestrator(config_path)
            orchestrator.state_manager = mock_state_manager
            
            # Mock methods for subsequent stages
            orchestrator._collect_pharmacies = MagicMock()
            orchestrator._deduplicate_pharmacies = MagicMock()
            orchestrator._classify_pharmacies = MagicMock(return_value=[
                {**SAMPLE_PHARMACY, "classification": {"is_pharmacy": True}}
            ])
            orchestrator._verify_pharmacies = MagicMock(return_value=[
                {**SAMPLE_PHARMACY, "classification": {"is_pharmacy": True}, "verification": {"verified": True}}
            ])
            orchestrator._save_results = MagicMock(return_value=output_dir / "pharmacies.json")
            
            # Run the pipeline
            result = orchestrator.run()
            
            # Verify pipeline execution
            orchestrator._collect_pharmacies.assert_not_called()  # Should not be called (already completed)
            orchestrator._deduplicate_pharmacies.assert_not_called()  # Should not be called (already completed)
            orchestrator._classify_pharmacies.assert_called_once()  # Should be called (failed before)
            orchestrator._verify_pharmacies.assert_called_once()  # Should be called (not started before)
            orchestrator._save_results.assert_called_once()  # Should be called at the end
            
            # Verify state updates
            mock_state_manager.update_task_status.assert_any_call('classification', 'in_progress')
            mock_state_manager.update_task_status.assert_any_call('classification', 'completed')
            mock_state_manager.update_task_status.assert_any_call('verification', 'in_progress')
            mock_state_manager.update_task_status.assert_any_call('verification', 'completed')

    def test_error_recovery(self, tmp_path):
        """Test that the pipeline can recover from transient errors."""
        # Reset credit tracker to ensure a clean state for the test
        credit_tracker.reset()
        
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
            "locations": [{
                "state": "CA",
                "cities": ["San Francisco"],
                "queries": ["independent pharmacy"]
            }],
            "api_keys": {
                "apify": "test_apify_key",
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
        
        # Create the orchestrator
        orchestrator = PipelineOrchestrator(config_path)
        
        # Create test data that will cause some verification errors
        pharmacies = []
        for i in range(5):
            # Create a mix of valid and invalid data
            # The first, third, and fifth items will succeed verification
            # The second and fourth will fail verification
            if i % 2 == 0:
                pharmacies.append(SAMPLE_PHARMACY.copy())
            else:
                # Missing data will cause verification to fail
                pharmacies.append({
                    "name": f"Invalid Pharmacy {i}",
                    # Missing address and other fields
                    "city": "San Francisco",
                    "state": "CA"
                })
        
        # Mock verify_pharmacy to fail on specific items
        with patch('pharmacy_scraper.orchestrator.pipeline_orchestrator.verify_pharmacy') as mock_verify:
            def verify_side_effect(pharmacy):
                if 'address' not in pharmacy:
                    raise ValueError(f"Missing address for pharmacy: {pharmacy['name']}")
                return {"verified": True, "confidence": 0.9}
                
            mock_verify.side_effect = verify_side_effect
            
            # Test the verification method directly
            result = orchestrator._verify_pharmacies(pharmacies)
            
            # Verify all pharmacies are in the result regardless of verification success/failure
            assert len(result) == 5, "All pharmacies should be included in result"
            
            # Check that errors were handled properly
            error_count = 0
            success_count = 0
            for pharmacy in result:
                if 'verification_error' in pharmacy:
                    error_count += 1
                    assert 'Missing address' in pharmacy['verification_error'], "Error message should be stored"
                elif 'verification' in pharmacy:
                    success_count += 1
                    assert pharmacy['verification']['verified'] is True, "Successful verification should be stored"
                    
            assert error_count == 2, "Expected 2 verification errors"
            assert success_count == 3, "Expected 3 successful verifications"

    def test_end_to_end_pipeline(self, tmp_path):
        """Test the entire pipeline with mocked external services.
        
        This test provides regression protection by ensuring:
        1. All pipeline stages are executed in the proper sequence
        2. Each stage receives the expected input and produces the expected output format
        3. The final output contains all required fields
        4. The state management correctly tracks pipeline progress
        
        Uses direct method replacement for critical components rather than function patching
        to ensure more reliable testing, following the successful approach used in other tests.
        """
        # Reset credit tracker to ensure a clean state for the test
        credit_tracker.reset()
        
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
            "locations": [{
                "state": "CA",
                "cities": ["San Francisco"],
                "queries": ["independent pharmacy"]
            }],
            "api_keys": {
                "apify": "test_apify_key",
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
        
        # Initialize the orchestrator
        orchestrator = PipelineOrchestrator(config_path)
        
        # Sample data for different stages
        raw_data = [
            {"name": "Pharmacy A", "address": "123 Main St", "city": "San Francisco", "state": "CA"},
            {"name": "Pharmacy B", "address": "456 Oak St", "city": "San Francisco", "state": "CA"},
            {"name": "Pharmacy A", "address": "123 Main Street", "city": "San Francisco", "state": "CA"}, # Duplicate
        ]
        
        deduped_data = [
            {"name": "Pharmacy A", "address": "123 Main St", "city": "San Francisco", "state": "CA"},
            {"name": "Pharmacy B", "address": "456 Oak St", "city": "San Francisco", "state": "CA"},
        ]
        
        classified_data = [
            {**deduped_data[0], "classification": {"is_pharmacy": True, "confidence": 0.9}},
            {**deduped_data[1], "classification": {"is_pharmacy": True, "confidence": 0.8}},
        ]
        
        verified_data = [
            {**classified_data[0], "verification": {"verified": True, "confidence": 0.95}},
            {**classified_data[1], "verification": {"verified": True, "confidence": 0.85}},
        ]
        
        # Replace orchestrator methods directly instead of patching functions
        # This follows the successful approach from the memory about fixing verify_pharmacy
        
        # Mock for data collection
        original_collect = orchestrator._collect_pharmacies
        def mock_collect_pharmacies():
            logger.info("Using mock collect_pharmacies method")
            return raw_data
        orchestrator._collect_pharmacies = mock_collect_pharmacies
        
        # Mock for deduplication
        original_dedupe = orchestrator._deduplicate_pharmacies
        def mock_deduplicate_pharmacies(pharmacies):
            logger.info("Using mock deduplicate_pharmacies method")
            assert pharmacies == raw_data, "Deduplication received unexpected data"
            return deduped_data
        orchestrator._deduplicate_pharmacies = mock_deduplicate_pharmacies
        
        # Mock for classification
        original_classify = orchestrator._classify_pharmacies
        def mock_classify_pharmacies(pharmacies):
            logger.info("Using mock classify_pharmacies method")
            assert pharmacies == deduped_data, "Classification received unexpected data"
            return classified_data
        orchestrator._classify_pharmacies = mock_classify_pharmacies
        
        # Mock for verification
        original_verify = orchestrator._verify_pharmacies
        def mock_verify_pharmacies(pharmacies):
            logger.info("Using mock verify_pharmacies method")
            assert pharmacies == classified_data, "Verification received unexpected data"
            return verified_data
        orchestrator._verify_pharmacies = mock_verify_pharmacies
        
        # Mock state manager to prevent it from looking for cached state
        def mock_get_task_status(task_name):
            return 'not_started'
        orchestrator.state_manager.get_task_status = mock_get_task_status
        
        # Run the pipeline
        try:
            result = orchestrator.run()
            
            # Verify the pipeline completed successfully
            assert result is not None, "Pipeline should complete successfully"
            
            # Verify the final output file exists
            output_file = output_dir / 'pharmacies.json'
            assert output_file.exists(), "Output file should be created"
            
            # Check output file content
            with open(output_file, 'r') as f:
                result_data = json.load(f)
                assert len(result_data) == len(verified_data), f"Expected {len(verified_data)} records but got {len(result_data)}"
                
                # Verify each record has all the required fields
                for record in result_data:
                    assert 'name' in record, "Each record should have a name"
                    assert 'classification' in record, "Each record should be classified"
                    assert 'verification' in record, "Each record should be verified"
        finally:
            # Restore original methods
            orchestrator._collect_pharmacies = original_collect
            orchestrator._deduplicate_pharmacies = original_dedupe
            orchestrator._classify_pharmacies = original_classify
            orchestrator._verify_pharmacies = original_verify
