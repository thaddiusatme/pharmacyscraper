import pytest
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.pharmacy_scraper.orchestrator.pipeline_orchestrator import PipelineOrchestrator, PipelineConfig
from src.pharmacy_scraper.orchestrator.state_manager import StateManager


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory for pipeline runs."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def mock_config(temp_output_dir):
    """Fixture for a mock PipelineConfig."""
    return PipelineConfig(
        api_keys={'apify': 'test_key', 'google_places': 'test_key'},
        output_dir=str(temp_output_dir),
        cache_dir=str(temp_output_dir / "cache"),
        verify_places=True,
        locations=[{'state': 'CA', 'cities': ['Test City']}]
    )


@pytest.fixture
def mock_state_manager(tmp_path):
    """Fixture for a mock StateManager."""
    db_path = tmp_path / "test_resume_state.db"
    return StateManager(db_path=str(db_path))


@pytest.fixture
def orchestrator(mock_config, mock_state_manager, temp_output_dir):
    """Fixture to create a PipelineOrchestrator instance with mocks."""
    # Create a dummy config file
    config_path = temp_output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump({
            'api_keys': mock_config.api_keys,
            'output_dir': mock_config.output_dir,
            'cache_dir': mock_config.cache_dir,
            'verify_places': mock_config.verify_places,
            'locations': mock_config.locations
        }, f)

    # The orchestrator will create its own StateManager, so we patch it
    with patch('src.pharmacy_scraper.orchestrator.pipeline_orchestrator.StateManager') as MockStateManager:
        MockStateManager.return_value = mock_state_manager
        orch = PipelineOrchestrator(config_path=str(config_path))
        # Mock the internal components to avoid actual API calls
        orch._collect_pharmacies = MagicMock(return_value=[{'id': 1, 'name': 'Test Pharmacy'}])
        orch._deduplicate_pharmacies = MagicMock(return_value=[{'id': 1, 'name': 'Test Pharmacy'}])
        orch._classify_pharmacies = MagicMock(return_value=[{'id': 1, 'name': 'Test Pharmacy', 'classification': 'independent'}])
        orch._verify_pharmacies = MagicMock(return_value=[{'id': 1, 'name': 'Test Pharmacy', 'classification': 'independent', 'verified': True}])
        yield orch


def test_full_pipeline_run_updates_state(orchestrator, mock_state_manager):
    """Test that a full, successful run marks all stages as completed."""
    # Run the pipeline
    orchestrator.run()

    # Assert that all stages were called and their status is 'completed'
    assert orchestrator._collect_pharmacies.called
    assert orchestrator._deduplicate_pharmacies.called
    assert orchestrator._classify_pharmacies.called
    assert orchestrator._verify_pharmacies.called

    for stage in orchestrator.STAGES:
        assert mock_state_manager.get_task_status(stage) == 'completed'


def test_resume_pipeline_skips_completed_stages(orchestrator, mock_state_manager, temp_output_dir):
    """Test that the pipeline resumes by skipping completed stages."""
    # Pre-populate state and create dummy output files for completed stages
    completed_stages = ['data_collection', 'deduplication']
    for stage in completed_stages:
        mock_state_manager.update_task_status(stage, 'completed')
        # Create dummy output file that the orchestrator expects to find
        stage_output_file = temp_output_dir / f"stage_{stage}_output.json"
        with open(stage_output_file, 'w') as f:
            json.dump([{'id': 1, 'name': 'Mock Data'}], f)

    # Run the pipeline
    orchestrator.run()

    # Assert that completed stages were NOT called again
    assert not orchestrator._collect_pharmacies.called
    assert not orchestrator._deduplicate_pharmacies.called

    # Assert that subsequent stages WERE called
    assert orchestrator._classify_pharmacies.called
    assert orchestrator._verify_pharmacies.called

    # Assert all stages are now marked as completed
    for stage in orchestrator.STAGES:
        assert mock_state_manager.get_task_status(stage) == 'completed'


def test_pipeline_stops_on_failed_stage(orchestrator, mock_state_manager):
    """Test that the pipeline halts if a stage fails."""
    # Mock a failure in the 'classification' stage
    error_message = "Classification API is down"
    orchestrator._classify_pharmacies.side_effect = Exception(error_message)

    # Run the pipeline
    result = orchestrator.run()

    # Assert the pipeline run failed
    assert result is None

    # Assert that stages before the failure are completed
    assert mock_state_manager.get_task_status('data_collection') == 'completed'
    assert mock_state_manager.get_task_status('deduplication') == 'completed'

    # Assert the failing stage is marked as 'failed'
    assert mock_state_manager.get_task_status('classification') == 'failed'

    # Assert that stages after the failure were not run
    assert not orchestrator._verify_pharmacies.called
    assert mock_state_manager.get_task_status('verification') is None
