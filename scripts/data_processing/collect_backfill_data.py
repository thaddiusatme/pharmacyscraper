import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Add project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.apify_collector import ApifyCollector
from src.utils.logger import get_file_logger

# Load environment variables from .env if present
load_dotenv()

def _clean_env_var(val: Optional[str]) -> Optional[str]:
    if val is None:
        return None
    return val.split("#")[0].strip()

def run_backfill():
    """Main function to run the backfill data collection."""
    logger = get_file_logger('collect_backfill_data')
    config_file = "config/backfill_run.json"
    logger.info(f"Starting data collection using config: {config_file}")

    # --- Start of explicit token configuration ---
    apify_api_token = _clean_env_var(os.getenv("APIFY_API_TOKEN"))
    apify_token = _clean_env_var(os.getenv("APIFY_TOKEN"))

    if apify_token and "${" in apify_token:
        final_token = apify_api_token
    else:
        final_token = apify_token or apify_api_token

    if not final_token:
        logger.error("Could not resolve Apify API token from .env file.")
        return
    # --- End of explicit token configuration ---

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # The backfill config directly contains the list of locations to query
        queries = config.get("locations", [])
        
        if not queries:
            logger.warning("No queries found in the config file. Exiting.")
            sys.exit(0)

        collector = ApifyCollector(api_token=final_token)
        logger.info(f"Starting data collection for {len(queries)} locations from config: {config_file}")
        
        collector.collect_pharmacies(queries)
        
        logger.info("Backfill data collection completed successfully.")

    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Failed to read or parse config file {config_file}: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred during data collection: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    run_backfill()
