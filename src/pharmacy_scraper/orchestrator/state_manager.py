import sqlite3
import logging

# Define the canonical pipeline stages in order of execution
PIPELINE_STAGES = [
    "data_collection",
    "data_classification",
    "data_processing",
    "data_verification",
    "deduplication_and_healing"
]


class StateManager:
    """Manages the state of the pipeline tasks using a SQLite database."""

    def __init__(self, db_path: str = 'pipeline_state.db'):
        """Initializes the StateManager and connects to the database."""
        self.db_path = db_path
        self._conn = None
        self._ensure_initialized()

    def _get_connection(self):
        """Returns a new database connection."""
        return sqlite3.connect(self.db_path)

    def _ensure_initialized(self):
        """Ensures the database and required tables exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pipeline_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_name TEXT UNIQUE NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('pending', 'in_progress', 'completed', 'failed')),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()

    def register_task(self, task_name: str):
        """Registers a new task with 'pending' status."""
        with self._get_connection() as conn:
            try:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO pipeline_tasks (task_name, status) VALUES (?, 'pending')", (task_name,))
                conn.commit()
            except sqlite3.IntegrityError:
                logging.warning(f"Task '{task_name}' is already registered.")

    def update_task_status(self, task_name: str, status: str):
        """Updates the status of a given task."""
        # First, register the task if it doesn't exist to ensure the record is there.
        current_status = self.get_task_status(task_name)
        if current_status is None:
            self.register_task(task_name)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE pipeline_tasks SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE task_name = ?", (status, task_name))
            conn.commit()

    def get_task_status(self, task_name: str) -> str | None:
        """Retrieves the status of a specific task."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT status FROM pipeline_tasks WHERE task_name = ?", (task_name,))
            result = cursor.fetchone()
            return result[0] if result else None

    def get_resume_point(self) -> str | None:
        """Determines which stage to resume the pipeline from."""
        for stage in PIPELINE_STAGES:
            status = self.get_task_status(stage)
            if status is None or status in ['pending', 'in_progress', 'failed']:
                return stage
        return None # All stages are completed

    def reset_state(self):
        """Clears all task records from the database for a fresh start."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM pipeline_tasks")
            conn.commit()
