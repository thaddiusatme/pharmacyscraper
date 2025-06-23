#!/usr/bin/env python3
"""
Resume pipeline script - handles partial runs and classification of existing data
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Set
import pandas as pd
import sys
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logger import get_file_logger

def load_config(config_path: str) -> Dict:
    """Load configuration file"""
    with open(config_path, 'r') as f:
        return json.load(f)

def find_existing_data(data_dir: str = "data") -> Dict[str, List[str]]:
    """Find existing data files and determine what states/cities have been collected"""
    existing_data = {}
    data_path = Path(data_dir)
    
    # Check various data directories
    for subdir in ["raw", "processed", "50_state_results", "pipeline_results"]:
        subdir_path = data_path / subdir
        if subdir_path.exists():
            for file_path in subdir_path.rglob("*.json"):
                # Extract state/city info from filename if possible
                filename = file_path.name
                if "pharmacies_" in filename:
                    parts = filename.replace("pharmacies_", "").replace(".json", "").split("_")
                    if len(parts) >= 2:
                        state = parts[0]
                        city = "_".join(parts[1:])
                        if state not in existing_data:
                            existing_data[state] = []
                        existing_data[state].append(city)
    
    return existing_data

def get_remaining_queries(config_path: str, existing_data: Dict[str, List[str]]) -> List[Dict]:
    """Determine which queries still need to be run"""
    config = load_config(config_path)
    remaining_queries = []
    
    for query_info in config.get("queries", []):
        state = query_info.get("state", "").upper()
        city = query_info.get("city", "").lower().replace(" ", "_")
        
        # Check if this state/city combo already exists
        if state not in existing_data or city not in existing_data[state]:
            remaining_queries.append(query_info)
    
    return remaining_queries

def create_resume_config(original_config_path: str, remaining_queries: List[Dict], output_path: str, logger=None):
    """Create a new config file with only the remaining queries"""
    if not logger:
        logger = logging.getLogger(__name__)

    original_config = load_config(original_config_path)
    
    resume_config = original_config.copy()
    resume_config["queries"] = remaining_queries
    resume_config["description"] = f"Resume run - {len(remaining_queries)} remaining queries"
    
    with open(output_path, 'w') as f:
        json.dump(resume_config, f, indent=2)
    
    logger.info(f"Created resume config with {len(remaining_queries)} queries: {output_path}")

def run_classification_on_existing_data(data_dir: str = "data", logger=None):
    """Run classification on any unclassified data found in data directories"""
    if not logger:
        logger = logging.getLogger(__name__)

    try:
        from run_pipeline import run_pipeline
    except ImportError:
        logger.error("Could not import 'run_pipeline'. Make sure that script exists and is in the python path.")
        return

    logger.info("Scanning for unclassified data...")
    data_path = Path(data_dir)
    
    # Look for JSON files that haven't been classified
    unclassified_files = []
    for subdir in ["raw", "50_state_results", "pipeline_results"]:
        subdir_path = data_path / subdir
        if subdir_path.exists():
            for file_path in subdir_path.rglob("*.json"):
                # Check if file contains unclassified pharmacy data
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list) and len(data) > 0:
                            # Check if any pharmacy lacks classification
                            sample_pharmacy = data[0]
                            if isinstance(sample_pharmacy, dict) and 'classification' not in sample_pharmacy:
                                unclassified_files.append(str(file_path))
                except Exception as e:
                    logger.warning(f"Could not read or parse {file_path}: {e}")
                    continue
    
    if unclassified_files:
        logger.info(f"Found {len(unclassified_files)} files with unclassified data")
        logger.info("Running classification on existing data...")
        
        # For each unclassified file, run classification
        for file_path in unclassified_files:
            logger.info(f"Classifying: {file_path}")
            # Here you would call your classification logic
            # This is a placeholder - you'd need to adapt based on your exact pipeline structure
    else:
        logger.info("No unclassified data found")

def main():
    parser = argparse.ArgumentParser(description="Resume or restart pipeline run")
    parser.add_argument("--config", required=True, help="Original config file path")
    parser.add_argument("--data-dir", default="data", help="Data directory to scan for existing results")
    parser.add_argument("--classify-existing", action="store_true", help="Run classification on existing unclassified data")
    parser.add_argument("--create-resume-config", help="Create resume config file path")
    parser.add_argument("--chunk-size", type=int, help="Split remaining queries into chunks of this size")
    
    args = parser.parse_args()
    logger = get_file_logger('resume_pipeline')
    
    if args.classify_existing:
        run_classification_on_existing_data(args.data_dir, logger=logger)
    
    # Find existing data
    existing_data = find_existing_data(args.data_dir)
    logger.info(f"Found existing data for {len(existing_data)} states:")
    for state, cities in existing_data.items():
        logger.info(f"  {state}: {len(cities)} cities")
    
    # Get remaining queries
    remaining_queries = get_remaining_queries(args.config, existing_data)
    logger.info(f"\nRemaining queries to run: {len(remaining_queries)}")
    
    if args.create_resume_config and remaining_queries:
        if args.chunk_size and len(remaining_queries) > args.chunk_size:
            # Create multiple smaller config files
            for i in range(0, len(remaining_queries), args.chunk_size):
                chunk = remaining_queries[i:i + args.chunk_size]
                chunk_config_path = args.create_resume_config.replace('.json', f'_chunk_{i//args.chunk_size + 1}.json')
                create_resume_config(args.config, chunk, chunk_config_path, logger=logger)
        else:
            create_resume_config(args.config, remaining_queries, args.create_resume_config, logger=logger)
    
    if not remaining_queries:
        logger.info("✅ All queries have been completed!")

if __name__ == "__main__":
    main()
