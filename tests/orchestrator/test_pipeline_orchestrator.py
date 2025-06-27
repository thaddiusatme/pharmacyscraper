import pytest
import pandas as pd
from types import SimpleNamespace
import json
from unittest.mock import MagicMock, call
import logging
from pathlib import Path

from src.pharmacy_scraper.orchestrator.pipeline_orchestrator import PipelineOrchestrator
from src.pharmacy_scraper.utils.api_usage_tracker import CreditLimitExceededError

class TestPipelineOrchestrator:
    """Test suite for the PipelineOrchestrator."""

    def test_execute_pharmacy_query_cache_miss(self, orchestrator_fixture: SimpleNamespace):
        """Test that a query is executed when there is a cache miss."""
        orchestrator = orchestrator_fixture.orchestrator
        mock_apify = orchestrator_fixture.mocks.apify
        mock_cache_load = orchestrator_fixture.mocks.cache_load
        mock_cache_save = orchestrator_fixture.mocks.cache_save

        mock_cache_load.return_value = None
        query = "test query"
        location = "Test City, CA"
        api_results = [{"name": "Test Pharmacy"}]
        mock_apify.run_trial.return_value = api_results

        result = orchestrator._execute_pharmacy_query(query, location)

        mock_cache_load.assert_called_once()
        mock_apify.run_trial.assert_called_once_with(query, location)
        mock_cache_save.assert_called_once()
        assert result == api_results

    def test_execute_pharmacy_query_cache_hit(self, orchestrator_fixture: SimpleNamespace):
        """Test that a query is not executed when there is a cache hit."""
        orchestrator = orchestrator_fixture.orchestrator
        mock_apify = orchestrator_fixture.mocks.apify
        mock_cache_load = orchestrator_fixture.mocks.cache_load

        query = "test query"
        location = "Test City, CA"
        cached_results = [{"name": "Cached Pharmacy"}]
        mock_cache_load.return_value = cached_results

        result = orchestrator._execute_pharmacy_query(query, location)

        mock_cache_load.assert_called_once()
        mock_apify.run_trial.assert_not_called()
        assert result == cached_results

    def test_verify_pharmacies_error_handling(self, orchestrator_fixture: SimpleNamespace):
        """Test error handling in the verification step."""
        orchestrator = orchestrator_fixture.orchestrator
        mock_verifier = orchestrator_fixture.mocks.verify_pharmacy
        
        pharmacies_df = pd.DataFrame([{"id": "1", "name": "Test Pharmacy"}])
        mock_verifier.side_effect = Exception("Verification failed")

        result_df = orchestrator._verify_pharmacies(pharmacies_df.to_dict('records'))
        
        assert "verification_error" in result_df[0]
        assert result_df[0]["verification_error"] == "Verification failed"

    def test_save_results(self, orchestrator_fixture: SimpleNamespace):
        """Test that results are saved correctly to JSON and CSV."""
        orchestrator = orchestrator_fixture.orchestrator
        pharmacies = [{"name": "Test Pharmacy", "city": "Testville"}]
        
        output_file = orchestrator._save_results(pharmacies)
        
        assert output_file.exists()
        assert output_file.name == "pharmacies.json"
        
        csv_file = output_file.parent / "pharmacies.csv"
        assert csv_file.exists()
        
        df = pd.read_csv(csv_file)
        assert df.iloc[0]["name"] == "Test Pharmacy"

    def test_run_pipeline(self, orchestrator_fixture: SimpleNamespace):
        """Test the full pipeline execution with realistic mock data."""
        orchestrator = orchestrator_fixture.orchestrator
        
        orchestrator_fixture.mocks.apify.run_trial.return_value = [
            {"title": "Test Pharmacy", "address": "123 Main St", "placeId": "p1"}
        ]
        orchestrator_fixture.mocks.remove_duplicates.return_value = pd.DataFrame([
            {"title": "Test Pharmacy", "address": "123 Main St", "placeId": "p1"}
        ])
        
        classification_result = SimpleNamespace()
        classification_result.to_dict = lambda: {"type": "independent", "confidence": 1.0}
        orchestrator_fixture.mocks.classifier.classify_pharmacy.return_value = classification_result
        
        orchestrator_fixture.mocks.verify_pharmacy.return_value = {
            "status": "VERIFIED", "confidence": 1.0, "verified_address": "123 Main St"
        }

        output_file = orchestrator.run()
        
        assert output_file is not None
        assert Path(output_file).exists()
        
        with open(output_file, 'r') as f:
            results = json.load(f)
            assert len(results) == 1
            assert results[0]['title'] == 'Test Pharmacy'
            assert results[0]['classification']['type'] == 'independent'
            assert results[0]['verification']['status'] == 'VERIFIED'

        orchestrator_fixture.mocks.apify.run_trial.assert_called()
        orchestrator_fixture.mocks.remove_duplicates.assert_called()
        orchestrator_fixture.mocks.classifier.classify_pharmacy.assert_called()
        orchestrator_fixture.mocks.verify_pharmacy.assert_called()

    def test_run_pipeline_credit_limit_exceeded(self, orchestrator_fixture: SimpleNamespace, caplog):
        """Test that the pipeline stops if the credit limit is exceeded."""
        orchestrator = orchestrator_fixture.orchestrator
        mock_apify = orchestrator_fixture.mocks.apify
        mock_apify.run_trial.side_effect = CreditLimitExceededError("APIs are too expensive!")

        with caplog.at_level(logging.ERROR):
            result = orchestrator.run()
            assert result is None
            assert "Budget exceeded: APIs are too expensive!" in caplog.text

    def test_run_pipeline_generic_exception(self, orchestrator_fixture: SimpleNamespace, caplog):
        """Test that the pipeline handles generic exceptions gracefully."""
        orchestrator = orchestrator_fixture.orchestrator
        mock_apify = orchestrator_fixture.mocks.apify
        mock_apify.run_trial.side_effect = Exception("A wild error appears!")

        with caplog.at_level(logging.ERROR):
            result = orchestrator.run()
            assert result is None
            assert "Pipeline failed: No pharmacies were collected. Check logs for details." in caplog.text

    def test_collect_pharmacies(self, orchestrator_fixture: SimpleNamespace):
        """Test the pharmacy collection process for a standard case."""
        orchestrator = orchestrator_fixture.orchestrator
        mock_apify = orchestrator_fixture.mocks.apify
        
        api_results = [{"name": "Test Pharmacy"}]
        mock_apify.run_trial.return_value = api_results
        
        results = orchestrator._collect_pharmacies()
        
        assert len(results) == 1
        assert results[0]['name'] == 'Test Pharmacy'
        mock_apify.run_trial.assert_called_once_with("test query", "Test City, CA")

    def test_run_pipeline_no_verification(self, orchestrator_fixture: SimpleNamespace):
        """Test pipeline execution when verification is disabled."""
        orchestrator = orchestrator_fixture.orchestrator
        orchestrator.config.verify_places = False

        # Mock the pipeline steps to avoid errors before the assertion
        orchestrator_fixture.mocks.apify.run_trial.return_value = [{"name": "Test Pharmacy"}]
        orchestrator_fixture.mocks.remove_duplicates.return_value = pd.DataFrame([{"name": "Test Pharmacy"}])
        classification_result = SimpleNamespace()
        classification_result.to_dict = lambda: {"type": "independent"}
        orchestrator_fixture.mocks.classifier.classify_pharmacy.return_value = classification_result
        
        orchestrator.run()
        
        orchestrator_fixture.mocks.verify_pharmacy.assert_not_called()

    def test_collect_pharmacies_with_query_error(self, orchestrator_fixture: SimpleNamespace, caplog):
        """Test that pharmacy collection continues even if one query fails."""
        orchestrator = orchestrator_fixture.orchestrator
        mock_apify = orchestrator_fixture.mocks.apify

        orchestrator.config.locations.append({
            "state": "TX",
            "cities": ["Houston"],
            "queries": ["failing query"]
        })

        mock_apify.run_trial.side_effect = [
            [{"name": "Good Pharmacy"}],
            Exception("API exploded")
        ]

        with caplog.at_level(logging.ERROR):
            results = orchestrator._collect_pharmacies()
            assert "Failed to collect pharmacies for 'failing query' in Houston, TX: API exploded" in caplog.text

        assert len(results) == 1
        assert results[0]['name'] == 'Good Pharmacy'

    def test_execute_pharmacy_query_corrupted_cache(self, orchestrator_fixture: SimpleNamespace, caplog):
        """Test that a corrupted cache file is handled correctly."""
        orchestrator = orchestrator_fixture.orchestrator
        mock_apify = orchestrator_fixture.mocks.apify
        mock_cache_load = orchestrator_fixture.mocks.cache_load
        
        mock_cache_load.side_effect = json.JSONDecodeError("Corrupted", "doc", 0)
        
        with caplog.at_level(logging.WARNING):
            orchestrator._execute_pharmacy_query("query", "location")
            assert "Failed to load cache" in caplog.text
            
        mock_apify.run_trial.assert_called_once()

    def test_execute_pharmacy_query_api_failure(self, orchestrator_fixture: SimpleNamespace):
        """Test that an API failure during query execution raises an exception."""
        orchestrator = orchestrator_fixture.orchestrator
        mock_apify = orchestrator_fixture.mocks.apify
        mock_cache_load = orchestrator_fixture.mocks.cache_load

        mock_cache_load.return_value = None
        mock_apify.run_trial.side_effect = Exception("API call failed")

        with pytest.raises(Exception, match="API call failed"):
            orchestrator._execute_pharmacy_query("query", "location")

    def test_classify_pharmacies_success(self, orchestrator_fixture: SimpleNamespace):
        """Test successful classification of pharmacies."""
        orchestrator = orchestrator_fixture.orchestrator
        mock_classifier = orchestrator_fixture.mocks.classifier
        
        pharmacies_df = pd.DataFrame([{"id": "1", "name": "Test Pharmacy"}])
        classification_result = MagicMock()
        classification_result.to_dict.return_value = {"type": "independent"}
        mock_classifier.classify_pharmacy.return_value = classification_result

        result_df = orchestrator._classify_pharmacies(pharmacies_df.to_dict('records'))
        
        assert "classification" in result_df[0]
        assert result_df[0]["classification"]["type"] == "independent"
        mock_classifier.classify_pharmacy.assert_called_once()

    def test_classify_pharmacies_failure(self, orchestrator_fixture: SimpleNamespace):
        """Test handling of a classification failure."""
        orchestrator = orchestrator_fixture.orchestrator
        mock_classifier = orchestrator_fixture.mocks.classifier

        pharmacies_df = pd.DataFrame([{"id": "1", "name": "Test Pharmacy"}])
        mock_classifier.classify_pharmacy.side_effect = Exception("Classification failed")

        result_df = orchestrator._classify_pharmacies(pharmacies_df.to_dict('records'))
        
        assert "classification_error" in result_df[0]
        assert result_df[0]["classification_error"] == "Classification failed"

    def test_verify_pharmacies_success(self, orchestrator_fixture: SimpleNamespace):
        """Test successful verification of pharmacies."""
        orchestrator = orchestrator_fixture.orchestrator
        mock_verifier = orchestrator_fixture.mocks.verify_pharmacy
        
        pharmacies_df = pd.DataFrame([{"id": "1", "name": "Test Pharmacy"}])
        verification_result = {"status": "VERIFIED"}
        mock_verifier.return_value = verification_result

        verified_df = orchestrator._verify_pharmacies(pharmacies_df.to_dict('records'))
        
        assert "verification" in verified_df[0]
        assert verified_df[0]['verification'] == verification_result
        mock_verifier.assert_called_once()

    def test_execute_pharmacy_query_saves_to_cache(self, orchestrator_fixture: SimpleNamespace):
        """Test that query results are saved to cache."""
        orchestrator = orchestrator_fixture.orchestrator
        mock_apify = orchestrator_fixture.mocks.apify
        mock_cache_save = orchestrator_fixture.mocks.cache_save
        mock_cache_load = orchestrator_fixture.mocks.cache_load

        mock_cache_load.return_value = None
        query = "caching query"
        location = "Caching Location"
        api_results = [{"name": "Data to be cached"}]
        mock_apify.run_trial.return_value = api_results

        orchestrator._execute_pharmacy_query(query, location)

        cache_key = f"{query}_{location}".lower().replace(" ", "_")
        mock_cache_save.assert_called_once_with(api_results, cache_key, orchestrator.config.cache_dir)

    def test_collect_pharmacies_error_handling(self, orchestrator_fixture: SimpleNamespace, caplog):
        """Test that _collect_pharmacies raises an error if no locations are configured."""
        orchestrator = orchestrator_fixture.orchestrator
        orchestrator.config.locations = []

        with pytest.raises(RuntimeError, match="No locations configured"):
            orchestrator._collect_pharmacies()

    def test_collect_pharmacies_no_cities_specified(self, orchestrator_fixture: SimpleNamespace, caplog):
        """Test that a state-level query is performed when no cities are specified."""
        orchestrator = orchestrator_fixture.orchestrator
        mock_apify = orchestrator_fixture.mocks.apify

        orchestrator.config.locations = [
            {
                "state": "CA",
                "queries": ["pharmacy"]
            }
        ]
        
        mock_apify.run_trial.return_value = [{"name": "State-level Pharmacy"}]

        with caplog.at_level(logging.WARNING):
            results = orchestrator._collect_pharmacies()
            assert "No cities specified for state CA, using state-level query" in caplog.text

        mock_apify.run_trial.assert_called_once_with("pharmacy", "CA")
        
        assert len(results) == 1
        assert results[0]['name'] == 'State-level Pharmacy'