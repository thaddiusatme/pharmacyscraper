#!/usr/bin/env python3
"""
Run a trial data collection for the pharmacy verification project.
"""
import os
import json
import logging
from pathlib import Path
from .apify_collector import ApifyCollector
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: Path):
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise

def parse_args():
    parser = argparse.ArgumentParser(description="Run a trial Apify collection")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=str, help="Path to config JSON file")
    group.add_argument("--query", type=str, help="Single search query string")
    parser.add_argument("--location", type=str, help="Location label when using --query (e.g. 'AZ')")
    parser.add_argument("--max-results", type=int, default=10, help="Max results per query/search")
    parser.add_argument("--output", type=str, default="data/trial_results", help="Output directory")
    return parser.parse_args()

def main():
    args = parse_args()

    collector = ApifyCollector()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.query:
        # Single query mode
        logger.info(f"Running single query: {args.query}")
        try:
            # Parse query to extract state and city info if possible
            # For now, create a simple config-like structure
            query_config = [{"query": args.query, "city": "Phoenix", "state": "AZ"}]
            results = collector.collect_pharmacies(query_config)
            outfile = output_dir / "single_query_results.json"
            with open(outfile, "w") as fh:
                json.dump(results, fh, indent=2)
            logger.info(f"Collected {len(results)} items; saved to {outfile}")
        except Exception as exc:
            logger.error(f"Error running single query: {exc}")
        return

    # Config mode
    config_path = Path(args.config)
    config = load_config(config_path)
    max_results = args.max_results

    all_results = []
    for state, queries in config.get("queries", {}).items():
        logger.info(f"Processing state: {state}")
        for q in queries:
            q_str = q["query"] if isinstance(q, dict) else q
            logger.info(f"Running query: {q_str}")
            try:
                res = collector.collect_pharmacies([q] if isinstance(q, dict) else [q_str])
                all_results.extend(res)
                outfile = output_dir / f"results_{state}.json"
                with open(outfile, "w") as fh:
                    json.dump(res, fh, indent=2)
            except Exception as exc:
                logger.error(f"Error processing query {q_str}: {exc}")

    if all_results:
        combined = output_dir / "combined_results.json"
        with open(combined, "w") as fh:
            json.dump(all_results, fh, indent=2)
        logger.info(f"Saved combined results to {combined}")


if __name__ == "__main__":
    main()
