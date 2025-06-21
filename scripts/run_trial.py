#!/usr/bin/env python3
"""
Run a trial data collection for the pharmacy verification project.
"""
import os
import json
import logging
from pathlib import Path
from apify_collector import ApifyCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path):
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise

def main():
    # Load configuration
    config_path = Path(__file__).parent.parent / 'config' / 'trial_config.json'
    config = load_config(config_path)
    
    # Initialize collector
    collector = ApifyCollector()
    
    # Create output directory if it doesn't exist
    output_dir = Path('data/trial_results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run collection for each query
    all_results = []
    for state, queries in config['queries'].items():
        logger.info(f"Processing state: {state}")
        for query in queries:
            logger.info(f"Running query: {query['query']}")
            try:
                results = collector.collect_pharmacies([query])
                all_results.extend(results)
                logger.info(f"Collected {len(results)} results")
                
                # Save results after each query
                output_file = output_dir / f"results_{state}_{query['city'].replace(' ', '_')}.json"
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Saved results to {output_file}")
                
            except Exception as e:
                logger.error(f"Error processing query {query['query']}: {e}")
    
    # Save combined results
    if all_results:
        combined_file = output_dir / 'combined_results.json'
        with open(combined_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Saved combined results to {combined_file}")
    
    logger.info("Trial run completed!")

if __name__ == "__main__":
    main()
