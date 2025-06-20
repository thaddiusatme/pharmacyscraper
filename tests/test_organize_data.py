#!/usr/bin/env python3
"""Tests for the organize_data module."""

import os
import shutil
import tempfile
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import logging

# Import the functions to test
from scripts.organize_data import (
    setup_directories,
    move_existing_files,
    combine_csv_files,
    main
)

# Test data
SAMPLE_CSV_1 = """name,address,city,state,phone,rating
Pharmacy One,123 Main St,Los Angeles,CA,555-1234,4.5
Pharmacy Two,456 Oak St,San Francisco,CA,555-5678,4.0
"""

SAMPLE_CSV_2 = """name,address,city,state,phone,rating
Pharmacy Three,789 Pine St,San Diego,CA,555-9012,4.2
Pharmacy One,123 Main St,Los Angeles,CA,555-1234,4.5
"""

@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing and clean up afterward."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create source and target directories
        source_dir = Path(temp_dir) / "source"
        target_dir = Path(temp_dir) / "target"
        
        source_dir.mkdir()
        target_dir.mkdir()
        
        # Create some test CSV files
        (source_dir / "pharmacies1.csv").write_text(SAMPLE_CSV_1)
        (source_dir / "pharmacies2.csv").write_text(SAMPLE_CSV_2)
        
        # Create a log directory
        log_dir = Path(temp_dir) / "logs"
        log_dir.mkdir()
        
        # Create a processed directory
        processed_dir = Path(temp_dir) / "processed"
        processed_dir.mkdir()
        
        yield {
            "temp_dir": temp_dir,
            "source_dir": source_dir,
            "target_dir": target_dir,
            "log_dir": log_dir,
            "processed_dir": processed_dir,
        }

def test_setup_directories():
    """Test that setup_directories creates required directories."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test directories
        test_dirs = [
            Path(temp_dir) / "test_dir1",
            Path(temp_dir) / "test_dir2" / "nested"
        ]
        
        # Mock the directory list to return our test directories
        with patch('scripts.organize_data.Path.mkdir') as mock_mkdir:
            setup_directories()
            
        # Check that mkdir was called with the right arguments
        assert mock_mkdir.call_count == 4  # 4 directories in the original function

def test_move_existing_files(temp_dirs):
    """Test moving existing CSV files to the target directory."""
    # Call the function
    moved_files = move_existing_files(
        source_dir=str(temp_dirs["source_dir"]),
        target_dir=str(temp_dirs["target_dir"])
    )
    
    # Check that files were moved
    assert len(moved_files) == 2
    assert (temp_dirs["target_dir"] / "pharmacies1.csv").exists()
    assert (temp_dirs["target_dir"] / "pharmacies2.csv").exists()
    
    # Check that source directory is empty
    assert len(list(temp_dirs["source_dir"].glob("*"))) == 0

def test_move_existing_files_duplicate_filenames(temp_dirs):
    """Test handling of duplicate filenames when moving files."""
    # Create a file with the same name in the target directory
    (temp_dirs["target_dir"] / "pharmacies1.csv").write_text("dummy")
    
    # Call the function
    moved_files = move_existing_files(
        source_dir=str(temp_dirs["source_dir"]),
        target_dir=str(temp_dirs["target_dir"])
    )
    
    # Check that files were moved with new names
    assert len(moved_files) == 2
    assert any("_moved.csv" in str(f) for f in moved_files)

def test_combine_csv_files(temp_dirs, caplog):
    """Test combining CSV files with deduplication."""
    # Set the log level to capture INFO messages
    caplog.set_level(logging.INFO)
    
    # Setup test files
    input_dir = temp_dirs["target_dir"]
    (input_dir / "file1.csv").write_text(SAMPLE_CSV_1)
    (input_dir / "file2.csv").write_text(SAMPLE_CSV_2)
    
    output_file = temp_dirs["processed_dir"] / "combined.csv"
    
    # Call the function
    result = combine_csv_files(
        source_dir=str(input_dir),
        output_file=str(output_file)
    )
    
    # Print all log messages for debugging
    print("\n=== Log Messages ===")
    for record in caplog.records:
        print(f"{record.levelname}: {record.message}")
    print("===================\n")
    
    # Check the result
    assert result == str(output_file)
    assert output_file.exists()
    
    # Check the combined data
    df = pd.read_csv(output_file)
    assert len(df) == 3  # Should have 3 unique rows (1 duplicate in the sample data)
    
    # Check that we have some log messages
    log_messages = [r.message for r in caplog.records]
    print(f"Looking for log message containing: Read {len(pd.read_csv(input_dir/'file1.csv'))} rows from file1.csv")
    print(f"Actual log messages: {log_messages}")
    
    # Check for the specific log message we're looking for
    expected_msg = f"Read {len(pd.read_csv(input_dir/'file1.csv'))} rows from file1.csv"
    assert any(expected_msg in msg for msg in log_messages), \
        f"Expected log message not found. Looking for: {expected_msg}"
    
    # Check for the duplicate removal message
    assert any("Removed 1 duplicate rows." in msg for msg in log_messages)
    
    # Check for the final save message
    assert any(f"Combined data saved to {output_file} with 3 unique rows." in msg for msg in log_messages)

def test_combine_csv_files_empty_dir(temp_dirs, caplog):
    """Test combining CSV files when directory is empty."""
    # Call with empty directory
    result = combine_csv_files(
        source_dir=str(temp_dirs["target_dir"]),
        output_file=str(temp_dirs["processed_dir"] / "empty.csv")
    )
    
    # Check that we got an empty string and logged a warning
    assert result == ""
    assert any("No CSV files found to combine" in r.message for r in caplog.records)

def test_combine_csv_files_invalid_file(temp_dirs, caplog):
    """Test handling of invalid CSV files."""
    # Create an invalid CSV file
    invalid_file = temp_dirs["target_dir"] / "invalid.csv"
    invalid_file.write_text("not,a,csv,file")
    
    # Call the function
    output_file = temp_dirs["processed_dir"] / "combined.csv"
    result = combine_csv_files(
        source_dir=str(temp_dirs["target_dir"]),
        output_file=str(output_file)
    )
    
    # Print all log messages for debugging
    print("\n=== Log Messages (Invalid File Test) ===")
    for record in caplog.records:
        print(f"{record.levelname}: {record.message}")
    print("======================================\n")
    
    # Check that we handled the error by returning the output file path
    # even if no data was written
    assert result == str(output_file)
    
    # Check that we logged the error
    log_messages = [r.message for r in caplog.records]
    print(f"Looking for log message containing: Error reading invalid.csv:")
    print(f"Log messages: {log_messages}")
    
    assert any("Error reading invalid.csv:" in msg for msg in log_messages)

def test_main_success(temp_dirs, monkeypatch, caplog):
    """Test the main function with successful execution."""
    # Set the log level to capture INFO messages
    caplog.set_level(logging.INFO)
    
    # Setup test environment
    monkeypatch.setattr('scripts.organize_data.setup_directories', lambda: None)
    
    # Mock the move_existing_files function
    mock_moved_files = ["file1.csv", "file2.csv"]
    monkeypatch.setattr(
        'scripts.organize_data.move_existing_files',
        lambda *args, **kwargs: mock_moved_files
    )
    
    # Mock the combine_csv_files function
    mock_output = "combined.csv"
    monkeypatch.setattr(
        'scripts.organize_data.combine_csv_files',
        lambda *args, **kwargs: mock_output
    )
    
    # Call the main function
    main()
    
    # Print all log messages for debugging
    print("\n=== Main Test Log Messages ===")
    for record in caplog.records:
        print(f"{record.levelname}: {record.message}")
    print("============================\n")
    
    # Check that we logged the success message
    log_messages = [r.message for r in caplog.records]
    print(f"Looking for log message containing: 'Combined data saved to combined.csv'")
    print(f"Actual log messages: {log_messages}")
    
    # Check for the specific log message we're looking for
    assert any("Combined data saved to combined.csv" in msg for msg in log_messages), \
        "Expected 'Combined data saved to combined.csv' log message not found"
    
    # Check for other expected log messages
    assert any("Successfully created combined file: combined.csv" in msg for msg in log_messages)
    assert any("Data organization complete!" in msg for msg in log_messages)

def test_main_exception_handling(monkeypatch, caplog):
    """Test that main function handles exceptions properly."""
    # Make setup_directories raise an exception
    def mock_setup_directories():
        raise Exception("Test error")
    
    monkeypatch.setattr('scripts.organize_data.setup_directories', mock_setup_directories)
    
    # Call the main function and check that it handles the exception
    with pytest.raises(Exception, match="Test error"):
        main()
    
    # Check that we logged the error
    assert "An error occurred" in caplog.text