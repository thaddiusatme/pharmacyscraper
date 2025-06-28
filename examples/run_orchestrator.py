#!/usr/bin/env python3
"""
Example script demonstrating how to use the PipelineOrchestrator.

This script shows how to initialize and run the orchestrator with a configuration file.
"""

import argparse
import logging
import os
from pathlib import Path

# Add project root to Python path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from pharmacy_scraper.orchestrator import PipelineOrchestrator

def setup_logging(log_level=logging.INFO):
    """Configure logging for the application."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('pipeline.log')
        ]
    )

def main():
    """Run the pipeline orchestrator with the provided configuration."""
    parser = argparse.ArgumentParser(description='Run the pharmacy scraper pipeline')
    parser.add_argument('--config', type=str, default='config/orchestrator_config.json',
                       help='Path to the configuration file')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize and run the orchestrator
        logger.info(f"Starting pharmacy scraper pipeline with config: {args.config}")
        orchestrator = PipelineOrchestrator(config_path=args.config)
        success = orchestrator.run()
        
        if success:
            logger.info("Pipeline completed successfully!")
        else:
            logger.error("Pipeline completed with errors")
            return 1
            
    except Exception as e:
        logger.exception("Pipeline failed with an unexpected error")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
