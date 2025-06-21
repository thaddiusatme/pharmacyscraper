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
from src.classification.classifier import classify_pharmacy

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


def _classify_batch(pharmacies: List[Dict], cache_dir: str | None = None) -> List[Dict]:
    """Annotate each *pharmacy* dict in-place with chain/independent labels."""
    processed: List[Dict] = []
    for p in pharmacies:
        try:
            res = classify_pharmacy(p, cache_dir=cache_dir)
            p.update(res)
            processed.append(p)
        except Exception as exc:  # noqa: BLE001
            logger.error("Classification failed for %s: %s", p.get("name"), exc)
            p.update({"is_chain": None, "confidence": 0.0, "method": "error"})
            processed.append(p)
    return processed


# ---------------------------------------------------------------------------
# Main pipeline logic
# ---------------------------------------------------------------------------

def run_pipeline(
    config_path: str,
    output_dir: str = "data/pipeline_results",
    budget: float = 100.0,
    classification_cache: str | None = ".api_cache/classify",
    skip_verification: bool = False,
) -> Path:
    """Execute the pipeline and write outputs under *output_dir*.

    Returns
    -------
    Path
        Path of the final CSV file for convenience.
    """

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
    collector = ApifyCollector(output_dir=output_dir)
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
    classified_records = _classify_batch(df_clean.to_dict(orient="records"), cache_dir=classification_cache)

    # ------------------------------------------------------------------
    # Phase 2b – Verification (Google Places)
    # ------------------------------------------------------------------
    if not skip_verification:
        logger.info("Running Google Places verification …")
        from src.verification.google_places import verify_batch
        verified_records = verify_batch(classified_records)
    else:
        logger.info("Skipping verification phase.")
        verified_records = classified_records

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_path = output_path / "pharmacies_final.csv"
    pd.DataFrame(verified_records).to_csv(csv_path, index=False)

    json_path = output_path / "pharmacies_final.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(verified_records, fh, ensure_ascii=False, indent=2)

    logger.info("✅ Pipeline completed – results written to %s", csv_path)

    return csv_path


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:  # noqa: D401 – simple helper
    """Return parsed CLI arguments."""
    parser = argparse.ArgumentParser(description="Run the full pharmacy pipeline")
    parser.add_argument("--config", required=True, help="Path to JSON configuration file")
    parser.add_argument("--output", default="data/pipeline_results", help="Output directory for results")
    parser.add_argument("--budget", type=float, default=100.0, help="Total credit budget (USD)")
    parser.add_argument("--cache_dir", default=".api_cache/classify", help="Cache directory for classification results")
    parser.add_argument("--skip_verify", action="store_true", help="Skip Google Places verification phase")
    return parser.parse_args()


def main() -> None:  # noqa: D401 – script entry-point
    args = _parse_args()
    try:
        run_pipeline(
            config_path=args.config,
            output_dir=args.output,
            budget=args.budget,
            classification_cache=args.cache_dir,
            skip_verification=args.skip_verify,
        )
    except CreditLimitExceededError as exc:
        logger.error("❌ %s", exc)
    except Exception as exc:  # noqa: BLE001
        logger.exception("❌ Pipeline failed: %s", exc)


if __name__ == "__main__":
    main()
