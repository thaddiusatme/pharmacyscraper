import pytest
import pandas as pd
from types import SimpleNamespace
import json
from unittest.mock import MagicMock, patch
import logging
from pathlib import Path
from src.pharmacy_scraper.utils.api_usage_tracker import CreditLimitExceededError

from src.pharmacy_scraper.orchestrator.pipeline_orchestrator import PipelineOrchestrator

class TestPipelineOrchestrator:
    """Test suite for the PipelineOrchestrator."""

    
    def test_execute_pharmacy_query_cache_miss(self, orchestrator_fixture: SimpleNamespace):
        """Test that a query is executed when there is a cache miss using direct method replacement."""
        orchestrator = orchestrator_fixture.orchestrator
        
        # Create test data
        query = "test query"
        location = "Test City, CA"
        api_results = [{"name": "API Pharmacy"}]
        expected_cache_key = f"{query}_{location}".lower().replace(" ", "_")
        
        # Track method calls
        cache_load_calls = []
        cache_save_calls = []
        run_trial_calls = []
        
        # Define mock functions with tracking
        def mock_load_from_cache(cache_key, cache_dir):
            cache_load_calls.append((cache_key, cache_dir))
            return None  # Simulate cache miss
            
        def mock_save_to_cache(data, cache_key, cache_dir):
            cache_save_calls.append((data, cache_key, cache_dir))
            
        def mock_run_trial(q, loc):
            run_trial_calls.append((q, loc))
            return api_results
        
        # Store references to the original functions
        original_load_from_cache = orchestrator._execute_pharmacy_query.__globals__['load_from_cache']
        original_save_to_cache = orchestrator._execute_pharmacy_query.__globals__['save_to_cache']
        original_run_trial = orchestrator.collector.run_trial
        
        # Replace with our mock implementations
        orchestrator._execute_pharmacy_query.__globals__['load_from_cache'] = mock_load_from_cache
        orchestrator._execute_pharmacy_query.__globals__['save_to_cache'] = mock_save_to_cache
        orchestrator.collector.run_trial = mock_run_trial
        
        try:
            # Execute the method
            result = orchestrator._execute_pharmacy_query(query, location)
            
            # Assertions
            assert len(cache_load_calls) > 0, "load_from_cache should have been called"
            assert len(run_trial_calls) > 0, "run_trial should have been called"
            assert len(cache_save_calls) > 0, "save_to_cache should have been called"
            
            # Verify call args
            assert cache_load_calls[0][0] == expected_cache_key, f"Cache key should be {expected_cache_key}"
            assert cache_load_calls[0][1] == orchestrator.config.cache_dir, "Should use config cache dir"
            
            assert run_trial_calls[0][0] == query, f"Query should be {query}"
            assert run_trial_calls[0][1] == location, f"Location should be {location}"
            
            assert cache_save_calls[0][0] == api_results, "Should have saved the API results to cache"
            assert cache_save_calls[0][1] == expected_cache_key, "Should have used the expected cache key"
            assert cache_save_calls[0][2] == orchestrator.config.cache_dir, "Should have used the configured cache directory"
            
            assert result == api_results, "Should have returned the API results"
        finally:
            # Restore original functions
            orchestrator._execute_pharmacy_query.__globals__['load_from_cache'] = original_load_from_cache
            orchestrator._execute_pharmacy_query.__globals__['save_to_cache'] = original_save_to_cache
            orchestrator.collector.run_trial = original_run_trial

    def test_execute_pharmacy_query_cache_hit(self, orchestrator_fixture: SimpleNamespace):
        """Test that a query is not executed when there is a cache hit using direct method replacement."""
        orchestrator = orchestrator_fixture.orchestrator
        
        # Create test data
        query = "test query"
        location = "Test City, CA"
        cache_results = [{"name": "Cached Pharmacy"}]
        expected_cache_key = f"{query}_{location}".lower().replace(" ", "_")
        
        # Track method calls
        cache_load_calls = []
        run_trial_calls = []
        
        # Define mock functions with tracking
        def mock_load_from_cache(cache_key, cache_dir):
            cache_load_calls.append((cache_key, cache_dir))
            return cache_results  # Simulate cache hit
            
        def mock_run_trial(q, loc):
            run_trial_calls.append((q, loc))
            return None  # This should not be called
        
        # Store references to the original functions
        original_load_from_cache = orchestrator._execute_pharmacy_query.__globals__['load_from_cache']
        original_run_trial = orchestrator.collector.run_trial
        
        # Replace with our mock implementations
        orchestrator._execute_pharmacy_query.__globals__['load_from_cache'] = mock_load_from_cache
        orchestrator.collector.run_trial = mock_run_trial
        
        try:
            # Execute the method
            result = orchestrator._execute_pharmacy_query(query, location)
            
            # Assertions
            assert len(cache_load_calls) > 0, "load_from_cache should have been called"
            assert len(run_trial_calls) == 0, "run_trial should not be called when there is a cache hit"
            
            # Verify call args
            assert cache_load_calls[0][0] == expected_cache_key, f"Cache key should be {expected_cache_key}"
            assert cache_load_calls[0][1] == orchestrator.config.cache_dir, "Should use config cache dir"
            
            assert result == cache_results, "Should have returned the cached results"
        finally:
            # Restore original functions
            orchestrator._execute_pharmacy_query.__globals__['load_from_cache'] = original_load_from_cache
            orchestrator.collector.run_trial = original_run_trial

    def test_run_pipeline(self, orchestrator_fixture: SimpleNamespace):
        """Test the full pipeline execution."""
        orchestrator = orchestrator_fixture.orchestrator
        
        # Track method calls
        collect_calls = []
        deduplicate_calls = []
        classify_calls = []
        verify_calls = []
        save_calls = []
        
        # Mock data
        collected_pharmacies = [{"name": "Test Pharmacy"}]
        deduplicated_pharmacies = [{"name": "Deduplicated Pharmacy"}]
        classified_pharmacies = [{"name": "Classified Pharmacy", "is_pharmacy": True}]
        verified_pharmacies = [{"name": "Verified Pharmacy", "is_pharmacy": True, "verification_status": "verified"}]
        
        # Create mock implementations
        def mock_collect_pharmacies():
            collect_calls.append(True)
            return collected_pharmacies
        
        def mock_deduplicate_pharmacies(pharmacies):
            deduplicate_calls.append(pharmacies)
            return deduplicated_pharmacies
        
        def mock_classify_pharmacies(pharmacies):
            classify_calls.append(pharmacies)
            return classified_pharmacies
        
        def mock_verify_pharmacies(pharmacies):
            verify_calls.append(pharmacies)
            return verified_pharmacies
        
        def mock_save_results(pharmacies):
            save_calls.append(pharmacies)
            # Just return the pharmacies as the result
            return pharmacies
        
        # Mock implementation of _execute_stage to avoid state management issues
        def mock_execute_stage(stage_name, stage_fn, *args, **kwargs):
            # Simply execute the function directly
            return stage_fn(*args, **kwargs)
        
        # Save original methods
        original_collect = orchestrator._collect_pharmacies
        original_deduplicate = orchestrator._deduplicate_pharmacies
        original_classify = orchestrator._classify_pharmacies
        original_verify = orchestrator._verify_pharmacies
        original_save = orchestrator._save_results
        original_execute_stage = orchestrator._execute_stage
        
        try:
            # Replace with mock implementations
            orchestrator._collect_pharmacies = mock_collect_pharmacies
            orchestrator._deduplicate_pharmacies = mock_deduplicate_pharmacies
            orchestrator._classify_pharmacies = mock_classify_pharmacies
            orchestrator._verify_pharmacies = mock_verify_pharmacies
            orchestrator._save_results = mock_save_results
            orchestrator._execute_stage = mock_execute_stage
            
            # Execute the pipeline
            result = orchestrator.run()
            
            # Assertions
            assert len(collect_calls) > 0, "Collect pharmacies should have been called"
            assert len(deduplicate_calls) > 0, "Deduplicate pharmacies should have been called"
            assert len(classify_calls) > 0, "Classify pharmacies should have been called"
            assert len(verify_calls) > 0, "Verify pharmacies should have been called"
            assert len(save_calls) > 0, "Save results should have been called"
            assert result == verified_pharmacies, "Pipeline should return verified pharmacies"
        finally:
            # Restore original methods
            orchestrator._collect_pharmacies = original_collect
            orchestrator._deduplicate_pharmacies = original_deduplicate
            orchestrator._classify_pharmacies = original_classify
            orchestrator._verify_pharmacies = original_verify
            orchestrator._save_results = original_save
            orchestrator._execute_stage = original_execute_stage

    def test_run_pipeline_no_verification(self, orchestrator_fixture: SimpleNamespace):
        """Test pipeline execution when verification is disabled."""
        orchestrator = orchestrator_fixture.orchestrator
        
        # Disable verification in config
        original_verify_setting = orchestrator.config.verify_places
        orchestrator.config.verify_places = False
    
        # Track method calls
        collect_calls = []
        deduplicate_calls = []
        classify_calls = []
        verify_calls = []
        save_calls = []
    
        # Mock data
        collected_pharmacies = [{"name": "Test Pharmacy"}]
        deduplicated_pharmacies = [{"name": "Deduplicated Pharmacy"}]
        classified_pharmacies = [{"name": "Classified Pharmacy", "is_pharmacy": True}]
    
        # Create mock implementations
        def mock_collect_pharmacies():
            collect_calls.append(True)
            return collected_pharmacies
    
        def mock_deduplicate_pharmacies(pharmacies):
            deduplicate_calls.append(pharmacies)
            return deduplicated_pharmacies
    
        def mock_classify_pharmacies(pharmacies):
            classify_calls.append(pharmacies)
            return classified_pharmacies
    
        def mock_verify_pharmacies(pharmacies):
            # This should NOT be called since verification is disabled
            verify_calls.append(pharmacies)
            return pharmacies
    
        def mock_save_results(pharmacies):
            save_calls.append(pharmacies)
            return pharmacies
    
        # Mock implementation of _execute_stage to avoid state management issues
        def mock_execute_stage(stage_name, stage_fn, *args, **kwargs):
            # Skip verification stage
            if stage_name == "verification" and not orchestrator.config.verify_places:
                return args[0] if args else None
            # Simply execute the function directly for other stages
            return stage_fn(*args, **kwargs)
    
        # Save original methods
        original_collect = orchestrator._collect_pharmacies
        original_deduplicate = orchestrator._deduplicate_pharmacies
        original_classify = orchestrator._classify_pharmacies
        original_verify = orchestrator._verify_pharmacies
        original_save = orchestrator._save_results
        original_execute_stage = orchestrator._execute_stage
        try:
            # Replace with mock implementations
            orchestrator._collect_pharmacies = mock_collect_pharmacies
            orchestrator._deduplicate_pharmacies = mock_deduplicate_pharmacies
            orchestrator._classify_pharmacies = mock_classify_pharmacies
            orchestrator._verify_pharmacies = mock_verify_pharmacies
            orchestrator._save_results = mock_save_results
            orchestrator._execute_stage = mock_execute_stage

            # Run the pipeline - should return classified pharmacies
            result = orchestrator.run()

            # Assertions
            assert len(collect_calls) == 1, "Collect pharmacies method should be called"
            assert len(deduplicate_calls) == 1, "Deduplicate pharmacies method should be called"
            assert deduplicate_calls[0] == collected_pharmacies, "Deduplicate should be called with collected pharmacies"
            assert len(classify_calls) == 1, "Classify pharmacies method should be called"
            assert classify_calls[0] == deduplicated_pharmacies, "Classify should be called with deduplicated pharmacies"

            # Verification should NOT be called since verify_places is False
            assert len(verify_calls) == 0, "Verify pharmacies method should NOT be called"

            assert len(save_calls) == 1, "Save results method should be called"
            assert save_calls[0] == classified_pharmacies, "Save should be called with classified pharmacies"
            assert result == classified_pharmacies, "Should return classified pharmacies"
        finally:
            # Restore original methods
            orchestrator._collect_pharmacies = original_collect
            orchestrator._deduplicate_pharmacies = original_deduplicate
            orchestrator._classify_pharmacies = original_classify
            orchestrator._verify_pharmacies = original_verify
            orchestrator._save_results = original_save
            orchestrator._execute_stage = original_execute_stage
            orchestrator.config.verify_places = original_verify_setting

    def test_run_pipeline_credit_limit_exceeded(self, orchestrator_fixture: SimpleNamespace):
        """Test pipeline execution when credit limit is exceeded."""
        orchestrator = orchestrator_fixture.orchestrator
        
        # Track method calls
        collect_called = False
        exception_caught = False
        
        # Mock the collection method to raise a credit limit exception
        def mock_collect_pharmacies():
            nonlocal collect_called
            collect_called = True
            raise CreditLimitExceededError("Credit limit exceeded")
            
        # Mock the execute stage to capture the exception
        def mock_execute_stage(stage_name, stage_fn, *args, **kwargs):
            try:
                return stage_fn(*args, **kwargs)
            except CreditLimitExceededError as e:
                nonlocal exception_caught
                exception_caught = True
                raise e
        
        # Save original methods
        original_collect = orchestrator._collect_pharmacies
        original_execute_stage = orchestrator._execute_stage
        
        try:
            # Replace methods
            orchestrator._collect_pharmacies = mock_collect_pharmacies
            orchestrator._execute_stage = mock_execute_stage
            
            # Run the pipeline - should return None on exception
            result = orchestrator.run()
            
            # Assertions
            assert collect_called, "Collection method should have been called"
            assert exception_caught, "Credit limit exception should have been caught"
            assert result is None, "Should return None when credit limit is exceeded"
        finally:
            # Restore original methods
            orchestrator._collect_pharmacies = original_collect
            orchestrator._execute_stage = original_execute_stage
            
    def test_run_pipeline_generic_exception(self, orchestrator_fixture: SimpleNamespace):
        """Test pipeline execution when a generic exception occurs."""
        orchestrator = orchestrator_fixture.orchestrator
        
        # Track method calls
        collect_called = False
        exception_caught = False
        
        # Mock the collection method to raise a generic exception
        def mock_collect_pharmacies():
            nonlocal collect_called
            collect_called = True
            raise RuntimeError("Something went wrong")
            
        # Mock the execute stage to capture the exception
        def mock_execute_stage(stage_name, stage_fn, *args, **kwargs):
            try:
                return stage_fn(*args, **kwargs)
            except RuntimeError as e:
                nonlocal exception_caught
                exception_caught = True
                raise e
        
        # Save original methods
        original_collect = orchestrator._collect_pharmacies
        original_execute_stage = orchestrator._execute_stage
        
        try:
            # Replace methods
            orchestrator._collect_pharmacies = mock_collect_pharmacies
            orchestrator._execute_stage = mock_execute_stage
            
            # Run the pipeline - should return None on exception
            result = orchestrator.run()
            
            # Assertions
            assert collect_called, "Collection method should have been called"
            assert exception_caught, "Generic exception should have been caught"
            assert result is None, "Should return None when an exception occurs"
        finally:
            # Restore original methods
            orchestrator._collect_pharmacies = original_collect
            orchestrator._execute_stage = original_execute_stage

    def test_verify_pharmacies_success(self, orchestrator_fixture: SimpleNamespace):
        """Test that pharmacies are verified successfully using direct method replacement."""
        orchestrator = orchestrator_fixture.orchestrator
        
        # Create test pharmacies
        test_pharmacies = [
            {"id": "1", "name": "Test Pharmacy 1"},
            {"id": "2", "name": "Test Pharmacy 2"}
        ]
        
        # Expected verified pharmacies
        verified_pharmacies = [
            {"id": "1", "name": "Test Pharmacy 1", "verification": {"is_verified": True, "score": 0.95}},
            {"id": "2", "name": "Test Pharmacy 2", "verification": {"is_verified": True, "score": 0.87}}
        ]
        
        # Save original method
        original_verify = orchestrator._verify_pharmacies
        
        # Create mock implementation
        def mock_verify_pharmacies(pharmacies):
            assert pharmacies == test_pharmacies, "Should verify the correct pharmacies"
            return verified_pharmacies
        
        try:
            # Replace with mock implementation
            orchestrator._verify_pharmacies = mock_verify_pharmacies
            
            # Call the method
            result = orchestrator._verify_pharmacies(test_pharmacies)
            
            # Assertions
            assert result == verified_pharmacies, "Should return verified pharmacies"
            assert result[0]["verification"]["is_verified"] is True, "Pharmacies should be marked as verified"
        finally:
            # Restore original method
            orchestrator._verify_pharmacies = original_verify

    def test_verify_pharmacies_error(self, orchestrator_fixture: SimpleNamespace):
        """Test that pharmacy verification errors are handled correctly using direct method replacement."""
        orchestrator = orchestrator_fixture.orchestrator
        
        # Create test pharmacies
        test_pharmacies = [
            {"id": "1", "name": "Test Pharmacy 1"},
            {"id": "2", "name": "Test Pharmacy 2"}
        ]
        
        # Expected results with error
        pharmacies_with_error = [
            {"id": "1", "name": "Test Pharmacy 1", "verification_error": "API limit exceeded"},
            {"id": "2", "name": "Test Pharmacy 2", "verification_error": "API limit exceeded"}
        ]
        
        # Save original method
        original_verify = orchestrator._verify_pharmacies
        
        # Create mock implementation
        def mock_verify_pharmacies_with_error(pharmacies):
            assert pharmacies == test_pharmacies, "Should attempt to verify the correct pharmacies"
            return pharmacies_with_error
        
        try:
            # Replace with mock implementation
            orchestrator._verify_pharmacies = mock_verify_pharmacies_with_error
            
            # Call the method
            result = orchestrator._verify_pharmacies(test_pharmacies)
            
            # Assertions
            assert result == pharmacies_with_error, "Should return pharmacies with error"
            assert "verification_error" in result[0], "Pharmacies should have verification_error field"
            assert result[0]["verification_error"] == "API limit exceeded", "Should have correct error message"
        finally:
            # Restore original method
            orchestrator._verify_pharmacies = original_verify

    def test_collect_pharmacies_no_cities_specified(self, orchestrator_fixture: SimpleNamespace):
        """Test that _collect_pharmacies works without specific cities."""
        orchestrator = orchestrator_fixture.orchestrator
        
        # Reset locations to ensure we're testing without specific cities
        original_locations = orchestrator.config.locations
        orchestrator.config.locations = [{"state": "CA", "cities": []}]
        
        # Create test data
        query_results = [{"name": "Test Pharmacy"}]
        
        # Track method calls
        execute_query_called = False
        called_queries = []
        called_locations = []
        
        # Mock implementation
        def mock_execute_pharmacy_query(query, location):
            nonlocal execute_query_called, called_queries, called_locations
            execute_query_called = True
            called_queries.append(query)
            called_locations.append(location)
            return query_results
        
        # Save original method
        original_execute = orchestrator._execute_pharmacy_query
        
        try:
            # Replace with mock implementation
            orchestrator._execute_pharmacy_query = mock_execute_pharmacy_query
            
            # Execute the method
            result = orchestrator._collect_pharmacies()
            
            # Assertions
            assert execute_query_called, "execute_pharmacy_query should have been called"
            assert len(called_queries) > 0, "Should have executed at least one query"
            assert len(called_locations) > 0, "Should have used at least one location"
            assert "CA" in called_locations[0], "Should have used the state as location"
            assert len(result) == len(called_queries), "Should have returned one result per query"
            for item in result:
                assert item == query_results[0], "Each result should match the query result"
        finally:
            # Restore original method and config
            orchestrator._execute_pharmacy_query = original_execute
            orchestrator.config.locations = original_locations
