#!/usr/bin/env python3
"""A dedicated script to run data collection for specified states and cities."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from scripts.apify_collector import ApifyCollector

# Load environment variables from .env if present
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    """Main function to run the data collection."""
    parser = argparse.ArgumentParser(description="Run Apify data collection for specified locations.")
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON config file.")
    args = parser.parse_args()

    # --- Start of explicit token configuration ---
    def _clean_env_var(val: Optional[str]) -> Optional[str]:
        if val is None:
            return None
        return val.split("#")[0].strip()

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
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Flatten the config into a list of queries, mirroring apify_collector's internal logic
        queries = []
        search_queries = config.get("search_queries", [])
        locations = config.get("locations", [])

        for loc in locations:
            for query_template in search_queries:
                queries.append(
                    {
                        "query": query_template["query"],
                        "max_results": query_template.get("max_results", 100),
                        "city": loc["city"],
                        "state": loc["state"],
                    }
                )
        
        if not queries:
            logger.warning("No queries generated from the config file. Exiting.")
            sys.exit(0)

        collector = ApifyCollector(api_token=final_token)
        logger.info(f"Starting data collection for {len(queries)} locations from config: {args.config}")
        
        # The collector expects a flat list of query dictionaries
        collector.collect_pharmacies(queries)
        
        logger.info("Data collection completed successfully.")

    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Failed to read or parse config file {args.config}: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred during data collection: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
