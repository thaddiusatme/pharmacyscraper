"""End-to-end orchestration script for the Independent Pharmacy Verification project.

This command-line utility stitches together the *collection → cleaning →
classification* phases into a single, reproducible pipeline.  It deliberately
covers **Phase 1** (Apify collection), **Phase 1.5** (deduplication) and
**Phase 2a** (chain vs independent classification).  The verification step
(Phase 2b – Google Places) is currently a **stub** and will be implemented in
future work.

Example
-------
$ export APIFY_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxx  # or set via .env
$ python -m scripts.run_pipeline --config config/trial_config.json \
                                  --output data/pipeline_results
"""
from __future__ import annotations

import argparse
from dotenv import load_dotenv
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd

# Phase 1 – data collection
from scripts.apify_collector import ApifyCollector

# Phase 1.5 – deduplication / self-healing utilities
from src.dedup_self_heal.dedup import remove_duplicates

# Phase 2 – classification utilities
from src.classification.classifier import Classifier
from src.classification.perplexity_client import PerplexityClient

# Budget / credit tracking
from src.utils.api_usage_tracker import credit_tracker, APICreditTracker, CreditLimitExceededError

logger = logging.getLogger("run_pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
# Load environment variables from .env if present
load_dotenv()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flatten_results(results: Union[List[Dict], Dict[str, List[Dict]]]) -> List[Dict]:
    """Convert nested *city → list[dict]* mapping into a single list."""
    if isinstance(results, list):
        return results

    flat: List[Dict] = []
    for city_items in results.values():
        flat.extend(city_items)
    return flat


def _classify_batch(
    pharmacies: List[Dict[str, Any]],
    cache_dir: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Classify a batch of pharmacies using the Classifier class."""
    try:
        client = PerplexityClient(cache_dir=cache_dir)
        classifier = Classifier(client)
    except ValueError as e:
        logger.error(f"Failed to initialize PerplexityClient: {e}")
        # Return empty results or raise, for now, let's log and exit
        sys.exit(1)

    classified_pharmacies = []
    for pharmacy in pharmacies:
        try:
            # The classification result is a dictionary
            result = classifier.classify_pharmacy(pharmacy)
            # Merge the result back into the original pharmacy data
            pharmacy.update(result)
            classified_pharmacies.append(pharmacy)
        except Exception as exc:
            logger.error(f"Classification failed for {pharmacy.get('title', 'N/A')}: {exc}")
            pharmacy.update({"is_chain": None, "confidence": 0.0, "method": "error"})
            classified_pharmacies.append(pharmacy)
            
    return classified_pharmacies


# ---------------------------------------------------------------------------
# Main pipeline logic
# ---------------------------------------------------------------------------

def run_pipeline(
    config_path: str,
    output_dir: str,
    budget: float,
    apify_cache_dir: str, 
    classification_cache_dir: str,
    deduplication_config_path: str,
    skip_verification: bool = False,
) -> Path:
    """Execute the pipeline and write outputs under *output_dir*.

    Returns
    -------
    Path
        Path of the final CSV file for convenience.
    """
    # Resolve config path to absolute to prevent file not found errors
    config_path = str(Path(config_path).resolve())

    # ------------------------------------------------------------------
    # Budget guardrail
    # ------------------------------------------------------------------
    tracker: APICreditTracker = credit_tracker  # global instance
    tracker.budget = budget  # ensure up-to-date total budget

    if not tracker.check_credit_available(estimated_cost=0):
        raise CreditLimitExceededError("Budget exhausted – aborting pipeline run.")

    # ------------------------------------------------------------------
    # Phase 1 – Collection
    # ------------------------------------------------------------------
    logger.info("Starting Phase 1 – data collection via Apify …")
    collector = ApifyCollector(output_dir=output_dir, cache_dir=apify_cache_dir)
    raw_results = collector.run_trial(config_path)
    pharmacies = _flatten_results(raw_results)
    if not pharmacies:
        raise RuntimeError("Apify returned no results – aborting.")
    logger.info("Collected %d raw pharmacy records", len(pharmacies))

    # ------------------------------------------------------------------
    # Phase 1.5 – Deduplication
    # ------------------------------------------------------------------
    df = pd.DataFrame(pharmacies)
    df_clean = remove_duplicates(df)
    logger.info("After deduplication: %d unique records (%.1f%% duplicates)",
                len(df_clean),
                100.0 * (1 - len(df_clean) / len(df)) if len(df) else 0.0)

    # ------------------------------------------------------------------
    # Phase 2 – Classification (chain vs independent)
    # ------------------------------------------------------------------
    logger.info("Running classification …")
    classification_results = _classify_batch(df_clean.to_dict(orient="records"), cache_dir=classification_cache_dir)

    # The classification_results now contain the merged data, so we just create a DataFrame
    df_classified = pd.DataFrame(classification_results)
    logger.info("Successfully merged classification results.")

    # ------------------------------------------------------------------
    # Phase 2b – Verification (Google Places)
    # ------------------------------------------------------------------
    if not skip_verification:
        logger.info("Running Google Places verification …")
        from src.verification.google_places import verify_batch
        # Pass the full classified records to the verification function
        verified_records = verify_batch(df_classified.to_dict(orient="records"))
    else:
        logger.info("Skipping verification phase.")
        # If skipping, the final records are the classified ones
        verified_records = df_classified.to_dict(orient="records")

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_path = output_path / "pharmacies_final.csv"
    df_final = pd.DataFrame(verified_records)

    # Check for and log duplicate columns before fixing
    duplicated_cols = df_final.columns[df_final.columns.duplicated()]
    if not duplicated_cols.empty:
        logger.warning(f"Found duplicate columns, which will be dropped: {duplicated_cols.tolist()}")

    # Remove duplicate columns, keeping the first occurrence
    df_final = df_final.loc[:, ~df_final.columns.duplicated()]

    logger.info(f"Final columns after deduplication: {df_final.columns.tolist()}")
    df_final.to_csv(csv_path, index=False, header=True)

    json_path = output_path / "pharmacies_final.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(verified_records, fh, ensure_ascii=False, indent=2)

    logger.info("✅ Pipeline completed – results written to %s", csv_path)

    return csv_path


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def setup_logging(log_file: str | None = None, log_level: str = "INFO"):
    """Configure root logger to output to console and optionally a file."""
    # Remove any existing handlers to avoid duplicates
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Basic console configuration
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode='a')
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        logger.info("Logging to file: %s", log_file)


def _parse_args() -> argparse.Namespace:  # noqa: D401 – simple helper
    """Return parsed CLI arguments."""
    parser = argparse.ArgumentParser(description="Run the full pharmacy pipeline")
    parser.add_argument("--config", required=True, help="Path to JSON configuration file")
    parser.add_argument('--deduplication-config-path', type=str, default='config/deduplication_config.json', help='Path to the deduplication config file.')
    parser.add_argument('--classification-cache-dir', type=str, default='data/cache/classification', help='Directory to store classification cache files.')
    parser.add_argument('--log-file', type=str, default=None, help='Path to the log file.')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Set the logging level.')
    parser.add_argument("--output", default="data/pipeline_results", help="Output directory for results")
    parser.add_argument("--budget", type=float, default=100.0, help="Total credit budget (USD)")

    parser.add_argument("--apify-cache-dir", default=".api_cache/apify", help="Cache directory for Apify results")
    parser.add_argument("--skip_verify", action="store_true", help="Skip Google Places verification phase")
    return parser.parse_args()


def main() -> None:  # noqa: D401 – script entry-point
    args = _parse_args()
    setup_logging(args.log_file, args.log_level)
    try:
        # Resolve all path arguments to absolute paths to prevent errors
        config_path_abs = str(Path(args.config).resolve())
        output_dir_abs = str(Path(args.output).resolve())
        dedup_config_path_abs = str(Path(args.deduplication_config_path).resolve())
        class_cache_dir_abs = str(Path(args.classification_cache_dir).resolve()) if args.classification_cache_dir else None
        apify_cache_dir_abs = str(Path(args.apify_cache_dir).resolve())

        run_pipeline(
            config_path=config_path_abs,
            output_dir=output_dir_abs,
            deduplication_config_path=dedup_config_path_abs,
            classification_cache_dir=class_cache_dir_abs,
            skip_verification=args.skip_verify,
            budget=args.budget,
            apify_cache_dir=apify_cache_dir_abs,
        )
    except CreditLimitExceededError as e:
        logger.error(f"❌ Pipeline stopped: {e}")
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
