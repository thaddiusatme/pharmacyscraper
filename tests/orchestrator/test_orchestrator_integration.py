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


def test_integration_full_run(orchestrator_with_stubs, temp_output_dir):
    """Test a full pipeline run with integrated (but stubbed) components."""
    orchestrator, stubs = orchestrator_with_stubs

    # Run the pipeline
    output_file = orchestrator.run()

    # Assertions
    assert output_file.exists()
    stubs['apify'].run_trial.assert_called_once()
    stubs['dedup'].assert_called_once()
    assert stubs['classifier'].classify_pharmacy.call_count == len(MOCK_DEDUPED_DATA)
    assert stubs['verifier'].call_count == len(MOCK_DEDUPED_DATA)

    # Check final output content
    with open(output_file, 'r') as f:
        final_data = json.load(f)
    assert final_data == MOCK_VERIFIED_DATA

    # Check state
    for stage in orchestrator.STAGES:
        assert orchestrator.state_manager.get_task_status(stage) == 'completed'


def test_integration_resume_run(orchestrator_with_stubs, temp_output_dir):
    """Test that a resumed run correctly skips completed stages."""
    orchestrator, stubs = orchestrator_with_stubs

    # --- First run: Simulate a failure during the first classification call ---
    stubs['classifier'].classify_pharmacy.side_effect = Exception("API Failure")
    first_run_output = orchestrator.run()
    assert first_run_output is None

    # Verify state after first run
    assert orchestrator.state_manager.get_task_status('data_collection') == 'completed'
    assert orchestrator.state_manager.get_task_status('deduplication') == 'completed'
    assert orchestrator.state_manager.get_task_status('classification') == 'failed'
    assert orchestrator.state_manager.get_task_status('verification') is None

    # --- Second run: Should resume and complete ---
    # Reset the mock's side effect to be successful
    mock_cr1 = MagicMock()
    mock_cr1.to_dict.return_value = MOCK_CLASSIFICATION_RESULTS[0]
    mock_cr2 = MagicMock()
    mock_cr2.to_dict.return_value = MOCK_CLASSIFICATION_RESULTS[1]
    stubs['classifier'].classify_pharmacy.side_effect = [mock_cr1, mock_cr2]

    second_run_output = orchestrator.run()

    # Assertions for the second run
    assert second_run_output.exists()
    # The first two stages should NOT have been called again
    stubs['apify'].run_trial.assert_called_once()
    stubs['dedup'].assert_called_once()

    # The failed stage should have been called again, and subsequent stages run
    # Called once in the failed run, and twice in the successful run
    assert stubs['classifier'].classify_pharmacy.call_count == 1 + len(MOCK_DEDUPED_DATA)
    assert stubs['verifier'].call_count == len(MOCK_DEDUPED_DATA)

    # Check final state
    for stage in orchestrator.STAGES:
        assert orchestrator.state_manager.get_task_status(stage) == 'completed'
