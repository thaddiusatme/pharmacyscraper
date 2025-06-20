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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/organize_data.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
            df = pd.read_csv(file)
            dfs.append(df)
            logger.info(f"Read {len(df)} rows from {file.name}")
        except Exception as e:
            logger.error(f"Error reading {file}: {str(e)}")
    
    if not dfs:
        logger.error("No valid CSV files could be read.")
        return ""
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Remove duplicates based on all columns
    initial_count = len(combined_df)
    combined_df = combined_df.drop_duplicates()
    removed_count = initial_count - len(combined_df)
    
    if removed_count > 0:
        logger.info(f"Removed {removed_count} duplicate rows.")
    
    # Save combined file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
            logger.info(f"Successfully created combined file: {combined_file}")
        
        logger.info("Data organization complete!")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
