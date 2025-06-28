import pytest
import sqlite3
from unittest.mock import patch, MagicMock

from src.pharmacy_scraper.orchestrator.state_manager import StateManager, PIPELINE_STAGES


@pytest.fixture
def state_manager(tmp_path):
    """Fixture to create a StateManager instance with a temporary db file."""
    db_path = tmp_path / "test_pipeline_state.db"
    return StateManager(db_path=str(db_path))


def test_initialization_creates_database_and_table(tmp_path):
    """Test that StateManager initializes the DB and the tasks table."""
    db_path = tmp_path / "test_init.db"
    assert not db_path.exists()

    sm = StateManager(db_path=str(db_path))

    assert db_path.exists()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='pipeline_tasks';")
        assert cursor.fetchone() is not None, "'pipeline_tasks' table should be created."

def test_register_task_adds_entry_to_db(state_manager):
    """Test that registering a task creates a new record in the database."""
    task_name = "collect_pharmacies_california"
    state_manager.register_task(task_name)

    with sqlite3.connect(state_manager.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM pipeline_tasks WHERE task_name = ?", (task_name,))
        task = cursor.fetchone()
        assert task is not None
        assert task[1] == task_name
        assert task[2] == 'pending', "Initial status should be 'pending'."

def test_update_task_status_modifies_record(state_manager):
    """Test that a task's status can be updated."""
    task_name = "classify_data_new_york"
    state_manager.register_task(task_name)
    state_manager.update_task_status(task_name, 'in_progress')

    with sqlite3.connect(state_manager.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT status FROM pipeline_tasks WHERE task_name = ?", (task_name,))
        status = cursor.fetchone()[0]
        assert status == 'in_progress'

    state_manager.update_task_status(task_name, 'completed')
    with sqlite3.connect(state_manager.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT status FROM pipeline_tasks WHERE task_name = ?", (task_name,))
        status = cursor.fetchone()[0]
        assert status == 'completed'

def test_get_task_status_retrieves_correct_status(state_manager):
    """Test retrieving the status of a specific task."""
    task_name = "process_results_texas"
    state_manager.register_task(task_name)
    assert state_manager.get_task_status(task_name) == 'pending'

    state_manager.update_task_status(task_name, 'completed')
    assert state_manager.get_task_status(task_name) == 'completed'

    assert state_manager.get_task_status('non_existent_task') is None

def test_get_resume_point_no_tasks(state_manager):
    """Test that the resume point is the first stage if no tasks are registered."""
    assert state_manager.get_resume_point() == PIPELINE_STAGES[0]

def test_get_resume_point_with_completed_tasks(state_manager):
    """Test that the resume point is the stage after the last completed one."""
    # Simulate completion of the first stage
    state_manager.update_task_status(PIPELINE_STAGES[0], 'completed')
    assert state_manager.get_resume_point() == PIPELINE_STAGES[1]

    # Simulate completion of the second stage as well
    state_manager.update_task_status(PIPELINE_STAGES[1], 'completed')
    assert state_manager.get_resume_point() == PIPELINE_STAGES[2]

def test_get_resume_point_with_all_tasks_completed(state_manager):
    """Test that if all stages are complete, the resume point is None."""
    for stage in PIPELINE_STAGES:
        state_manager.update_task_status(stage, 'completed')
    assert state_manager.get_resume_point() is None, "Should return None if all stages are completed."

def test_get_resume_point_with_in_progress_task(state_manager):
    """Test that an 'in_progress' task is identified as the resume point."""
    state_manager.update_task_status(PIPELINE_STAGES[0], 'completed')
    state_manager.update_task_status(PIPELINE_STAGES[1], 'in_progress')
    assert state_manager.get_resume_point() == PIPELINE_STAGES[1]

def test_reset_state_clears_all_tasks(state_manager):
    """Test that the state can be reset, clearing all task records."""
    state_manager.register_task("task1")
    state_manager.register_task("task2")
    state_manager.reset_state()

    with sqlite3.connect(state_manager.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM pipeline_tasks")
        count = cursor.fetchone()[0]
        assert count == 0, "The tasks table should be empty after reset."
