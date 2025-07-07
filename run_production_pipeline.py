#!/usr/bin/env python
"""
Production runner for the pharmacy scraper pipeline.
Securely loads environment variables from .env file and runs the pipeline.

Usage:
    python run_production_pipeline.py [--config CONFIG_PATH] [--reset]
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Import dotenv at the top to ensure environment variables are loaded before any other imports
try:
    from dotenv import load_dotenv
    # Load environment variables from .env file
    load_dotenv()
except ImportError:
    print("Error: python-dotenv package is not installed.")
    print("Please install it using: pip install python-dotenv")
    sys.exit(1)

# Now import the pipeline components
from src.pharmacy_scraper.orchestrator.pipeline_orchestrator import PipelineOrchestrator
from src.pharmacy_scraper.utils.api_usage_tracker import credit_tracker

# Configure logging with the project's preferred format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)8s] %(message)s (%(filename)s:%(lineno)d)",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def validate_environment() -> bool:
    """
    Validate that required environment variables are set.
    
    Returns:
        bool: True if all required environment variables are set, False otherwise.
    """
    required_vars = {
        "APIFY_TOKEN": os.getenv("APIFY_TOKEN") or os.getenv("APIFY_API_TOKEN"),
        "GOOGLE_MAPS_API_KEY": os.getenv("GOOGLE_MAPS_API_KEY"),
        "PERPLEXITY_API_KEY": os.getenv("PERPLEXITY_API_KEY")
    }
    
    missing = [key for key, value in required_vars.items() if not value]
    
    if missing:
        logger.error(f"Missing required environment variables: {', '.join(missing)}")
        logger.error("Please ensure these are set in your .env file")
        return False
        
    return True


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run the pharmacy scraper production pipeline.")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/production/secure_production_config.json",
        help="Path to the JSON configuration file for the pipeline."
    )
    parser.add_argument(
        "--reset", 
        action="store_true",
        help="Reset the pipeline state and start fresh."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate setup without making API calls."
    )
    
    return parser.parse_args()


def prepare_directories(config_path: str) -> None:
    """
    Ensure necessary directories exist based on the configuration.
    
    Args:
        config_path: Path to the configuration file
    """
    import json
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Create cache and output directories if they don't exist
        for dir_key in ['cache_dir', 'output_dir']:
            if dir_key in config:
                Path(config[dir_key]).mkdir(parents=True, exist_ok=True)
                logger.info(f"Ensured directory exists: {config[dir_key]}")
    
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(f"Error reading config file: {e}")
        sys.exit(1)


def run_pipeline(config_path: str, reset: bool = False, dry_run: bool = False) -> Optional[str]:
    """
    Run the pharmacy scraper pipeline.
    
    Args:
        config_path: Path to the configuration file
        reset: Whether to reset the pipeline state
        dry_run: Whether to perform a dry run without making API calls
    
    Returns:
        Optional[str]: Path to the output file if successful, None otherwise
    """
    try:
        # Initialize orchestrator
        orchestrator = PipelineOrchestrator(config_path=config_path)
        
        if reset:
            logger.info("Resetting pipeline state as requested")
            orchestrator.state_manager.reset_state()
            
        if dry_run:
            logger.info("Dry run mode - validating configuration only")
            logger.info(f"Configuration loaded successfully from: {config_path}")
            logger.info(f"Pipeline would run with the following stages:")
            logger.info(f"  - Data collection (locations: {len(orchestrator.config.locations)})")
            logger.info(f"  - Deduplication")
            logger.info(f"  - Classification")
            logger.info(f"  - Verification")
            logger.info(f"API budget: ${orchestrator.config.max_budget}")
            return None
            
        # Run the pipeline
        logger.info(f"Starting pipeline with configuration: {config_path}")
        result = orchestrator.run()
        
        if result:
            logger.info(f"✅ Pipeline completed successfully!")
            logger.info(f"Output saved to: {result}")
            return result
        else:
            logger.warning("❌ Pipeline did not complete successfully")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error running pipeline: {e}", exc_info=True)
        return None


def main() -> None:
    """Main entry point for the production pipeline runner."""
    args = parse_args()
    
    if not validate_environment():
        sys.exit(1)
        
    prepare_directories(args.config)
    
    credit_tracker.reset()
    logger.info("API credit tracker reset")
    
    result = run_pipeline(args.config, args.reset, args.dry_run)
    
    if result:
        try:
            import json
            with open(result, 'r') as f:
                data = json.load(f)
                logger.info(f"Processed {len(data)} pharmacies successfully")
        except Exception:
            pass
    
    # Return appropriate exit code
    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()
