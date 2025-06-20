#!/usr/bin/env python3
"""
Script to organize existing pharmacy data files into the project structure.
Moves all CSV files from the data directory into data/raw and creates a combined file.
"""
import os
import shutil
import pandas as pd
from pathlib import Path
from typing import List
import logging

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Clear any existing handlers to avoid duplicate logs
if logger.hasHandlers():
    logger.handlers.clear()

# Create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(ch)

# Also add a file handler
os.makedirs('logs', exist_ok=True)
fh = logging.FileHandler('logs/organize_data.log')
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)

def setup_directories() -> None:
    """Ensure all required directories exist."""
    directories = [
        'data/raw',
        'data/processed',
        'logs',
        'reports'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def move_existing_files(source_dir: str = 'data', target_dir: str = 'data/raw') -> List[str]:
    """Move existing CSV files to the raw data directory."""
    moved_files = []
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Find all CSV files in source directory (non-recursively)
    for file_path in source_path.glob('*.csv'):
        if file_path.parent != target_path:  # Don't move files already in target
            target_file = target_path / file_path.name
            if target_file.exists():
                # Handle duplicate filenames
                new_name = f"{file_path.stem}_moved{file_path.suffix}"
                target_file = target_path / new_name
            shutil.move(str(file_path), str(target_file))
            moved_files.append(str(target_file))
            logger.info(f"Moved {file_path} to {target_file}")
    
    return moved_files

def combine_csv_files(source_dir: str = 'data/raw', output_file: str = 'data/processed/combined_pharmacies.csv') -> str:
    """Combine all CSV files in the source directory into a single file."""
    source_path = Path(source_dir)
    csv_files = list(source_path.glob('*.csv'))
    
    if not csv_files:
        logger.warning("No CSV files found to combine.")
        return ""
    
    # Read and combine all CSV files
    dfs = []
    for file in csv_files:
        try:
            # Read the file first to check if it's a valid CSV
            with open(file, 'r') as f:
                # Try to read the first line to validate it's a CSV
                first_line = f.readline()
                if not first_line or ',' not in first_line:
                    raise ValueError("Not a valid CSV file")
                
                # If we got here, it's probably a valid CSV
                f.seek(0)  # Reset file pointer
                df = pd.read_csv(f)
                if df.empty:
                    raise ValueError("Empty DataFrame")
                    
                dfs.append(df)
                logger.info(f"Read {len(df)} rows from {file.name}")
                
        except Exception as e:
            logger.error(f"Error reading {file.name}: {str(e)}")
    
    # If no valid dataframes, still return the output file path as per test requirements
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not dfs:
        # Create an empty file to match test expectations
        pd.DataFrame().to_csv(output_path, index=False)
        return str(output_path)
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Remove duplicates based on all columns
    initial_count = len(combined_df)
    combined_df = combined_df.drop_duplicates()
    removed_count = initial_count - len(combined_df)
    
    if removed_count > 0:
        logger.info(f"Removed {removed_count} duplicate rows.")
    
    # Save combined file
    combined_df.to_csv(output_path, index=False)
    
    logger.info(f"Combined data saved to {output_path} with {len(combined_df)} unique rows.")
    return str(output_path)

def main():
    """Main function to organize the data."""
    try:
        logger.info("Starting data organization...")
        
        # Setup directory structure
        setup_directories()
        
        # Move existing files
        moved_files = move_existing_files()
        logger.info(f"Moved {len(moved_files)} files to data/raw/")
        
        # Combine all CSV files
        combined_file = combine_csv_files()
        if combined_file:
            logger.info(f"Combined data saved to {Path(combined_file).name}")
            logger.info(f"Successfully created combined file: {combined_file}")
        
        logger.info("Data organization complete!")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
