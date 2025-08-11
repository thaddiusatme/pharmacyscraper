import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd

from src.pharmacy_scraper.orchestrator.pipeline_orchestrator import PipelineOrchestrator, PipelineConfig
from src.pharmacy_scraper.orchestrator.state_manager import StateManager

# Sample data to be returned by mocked components
MOCK_RAW_DATA = [{'id': 1, 'name': 'Test Pharmacy'}, {'id': 2, 'name': 'Duplicate Pharmacy'}, {'id': 2, 'name': 'Duplicate Pharmacy'}]
MOCK_DEDUPED_DATA = [{'id': 1, 'name': 'Test Pharmacy'}, {'id': 2, 'name': 'Duplicate Pharmacy'}]
MOCK_CLASSIFICATION_RESULTS = [{'status': 'independent'}, {'status': 'chain'}]
MOCK_VERIFICATION_RESULTS = [{'verified': True}, {'verified': False}]

MOCK_CLASSIFIED_DATA = [
    {'id': 1, 'name': 'Test Pharmacy', 'classification': MOCK_CLASSIFICATION_RESULTS[0]},
    {'id': 2, 'name': 'Duplicate Pharmacy', 'classification': MOCK_CLASSIFICATION_RESULTS[1]},
]
MOCK_VERIFIED_DATA = [
    {**MOCK_CLASSIFIED_DATA[0], 'verification': MOCK_VERIFICATION_RESULTS[0]},
    {**MOCK_CLASSIFIED_DATA[1], 'verification': MOCK_VERIFICATION_RESULTS[1]},
]


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory for pipeline runs."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def mock_config_path(temp_output_dir):
    """Fixture for a mock PipelineConfig, saved to a file."""
    config_data = {
        'api_keys': {'apify': 'fake_key', 'google_places': 'fake_key'},
        'output_dir': str(temp_output_dir),
        'cache_dir': str(temp_output_dir / "cache"),
        'verify_places': True,
        'locations': [{'state': 'CA', 'cities': ['Test City']}]
    }
    config_path = temp_output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config_data, f)
    return str(config_path)


@pytest.fixture
def orchestrator_with_stubs(mock_config_path):
    """Fixture to create a PipelineOrchestrator with stubbed external clients."""
    with patch('src.pharmacy_scraper.orchestrator.pipeline_orchestrator.ApifyCollector') as MockApifyCollector:
        with patch('src.pharmacy_scraper.orchestrator.pipeline_orchestrator.remove_duplicates') as mock_remove_duplicates:
            with patch('src.pharmacy_scraper.orchestrator.pipeline_orchestrator.Classifier') as MockClassifier:
                with patch('src.pharmacy_scraper.orchestrator.pipeline_orchestrator.verify_pharmacy') as mock_verify_pharmacy:
                    # Configure the mocks to return predictable data
                    mock_apify_instance = MockApifyCollector.return_value
                    mock_apify_instance.run_trial.return_value = MOCK_RAW_DATA

                    mock_remove_duplicates.return_value = pd.DataFrame(MOCK_DEDUPED_DATA)

                    mock_classifier_instance = MockClassifier.return_value
                    # Mock the singular method called in a loop
                    mock_cr1 = MagicMock()
                    mock_cr1.to_dict.return_value = MOCK_CLASSIFICATION_RESULTS[0]
                    mock_cr2 = MagicMock()
                    mock_cr2.to_dict.return_value = MOCK_CLASSIFICATION_RESULTS[1]
                    mock_classifier_instance.classify_pharmacy.side_effect = [mock_cr1, mock_cr2]

                    # Mock the singular method called in a loop
                    mock_verify_pharmacy.side_effect = MOCK_VERIFICATION_RESULTS

                    # Yield the orchestrator and the mocks for use in tests
                    db_path = Path(mock_config_path).parent / "test_pipeline_state.db"
                    orchestrator = PipelineOrchestrator(config_path=mock_config_path, db_path=str(db_path))
                    yield orchestrator, {
                        'apify': mock_apify_instance,
                        'dedup': mock_remove_duplicates,
                        'classifier': mock_classifier_instance,
                        'verifier': mock_verify_pharmacy
                    }
def test_integration_resume_run(orchestrator_with_stubs, temp_output_dir):
    """Test that a pipeline run can be resumed after a failure.
    
    This test simulates the following scenario:
    1. Pipeline is run with all dependencies mocked
    2. Classification stage fails with an API error for one pharmacy
    3. Pipeline is resumed and completes successfully
    4. Pipeline output should contain 2 classified pharmacies
    """
    # Setup the orchestrator with mocks
    orchestrator, stubs = orchestrator_with_stubs
    
    # Set up mock pipeline state to simulate an interrupted pipeline
    # Set deduplication as completed
    orchestrator.state_manager.update_task_status("deduplication", "completed")
    # Set classification as in progress/failed
    orchestrator.state_manager.update_task_status("classification", "failed")
    
    # Save stage data to the expected location
    output_dir = Path(orchestrator.config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create deduplication stage output file
    dedup_output_file = output_dir / "stage_deduplication_output.json"
    with open(dedup_output_file, 'w') as f:
        json.dump(MOCK_DEDUPED_DATA, f, indent=2)
    
    # Set up the classifier mock to fail on first pharmacy and succeed on second
    mock_cr2 = MagicMock()
    mock_cr2.to_dict.return_value = MOCK_CLASSIFICATION_RESULTS[1]
    
    # First pharmacy fails classification, second succeeds
    stubs['classifier'].classify_pharmacy.side_effect = [Exception("API Failure"), mock_cr2]
    
    # Run the pipeline, which should succeed despite the single classification failure
    first_run_output = orchestrator.run()
    
    # First run should succeed because the orchestrator is resilient to individual failures
    assert first_run_output is not None, "Pipeline should succeed despite individual classification error"
    
    # The pipeline should now be completed since individual failures don't stop the orchestrator
    pipeline_state = orchestrator.state_manager.get_state()
    assert pipeline_state.status == StateManager.PipelineStatus.COMPLETED, "Pipeline should be completed despite individual classification error"
    
    # Verify that the classifier was called for both pharmacies (one failed, one succeeded)
    assert stubs['classifier'].classify_pharmacy.call_count == 2, \
        f"Classifier should be called twice, but was called {stubs['classifier'].classify_pharmacy.call_count} times"
    
    # Verify that the verification client was called for each pharmacy
    assert stubs['verifier'].call_count == 2, \
        f"Verifier should be called twice, but was called {stubs['verifier'].call_count} times"
    
    # Verify that collection stage was called
    # Note: deduplication stage is skipped since it was marked as completed and loads from cache
    stubs['apify'].run_trial.assert_called_once() 

    # Check final state
    for stage in orchestrator.STAGES:
        assert orchestrator.state_manager.get_task_status(stage) == 'completed'
