#!/usr/bin/env python3
"""verify_with_sonar_pro.py

One-off helper to validate a random 20 % sample of pharmacies using
Perplexity's **sonar-pro** (default) or **sonar** model.

It can either:
1. Sample rows from an Excel spreadsheet (the original workflow), **or**
2. Re-classify an *existing* CSV of rows (e.g. the output of the sonar-pro
   run) so you can compare different models on the **exact same items**.

Usage examples:

Re-sample 20 % of an Excel sheet and classify with *sonar-pro*:

    python scripts/verify_with_sonar_pro.py \
        --excel independent_pharmacies_20250623_094553.xlsx \
        --sample-fraction 0.2 \
        --batch-size 10

Re-classify the **same rows** that were already processed by sonar-pro,
this time with the *sonar* model:

    python scripts/verify_with_sonar_pro.py \
        --input-csv results/sonar_pro_trial.csv \
        --model sonar \
        --output results/sonar_validation.csv

The script:
1. Loads the spreadsheet with *pandas*.
2. Randomly samples the requested fraction (default 20 %).
3. Iterates over the sample in batches of *batch_size* (default 10).
4. Calls `PerplexityClient` with `model_name="sonar-pro"` to classify each
   pharmacy, collecting `classification`, `is_independent`, `confidence`, and
   `explanation`.
5. Saves a CSV (or JSON, if the filename ends with `.json`) containing the
   combined original row data plus classification fields.

Environment:
• Requires `PERPLEXITY_API_KEY` to be set.
• Make sure `pip install pandas openpyxl` is available (used by pandas to read
  XLSX files).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Add repo root to import path when executed as a script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.classification.perplexity_client import PerplexityClient  # noqa: E402


###############################################################################
# Helpers
###############################################################################


def _row_to_pharmacy(row: pd.Series) -> Dict:
    """Convert a DataFrame row to the dict expected by `PerplexityClient`.

    The spreadsheet column names may differ; adjust the mapping below if
    necessary.  We look for a few common variants to be resilient.
    """
    # Normalise column access (case-insensitive)
    lower_cols = {str(c).lower(): c for c in row.index}

    def _get(*possible: str, default: str = "") -> str:
        for p in possible:
            if p.lower() in lower_cols:
                return str(row[lower_cols[p.lower()]]).strip()
        return default

    return {
        "name": _get("name", "title"),
        "address": _get("address", "full_address", "street"),
        "phone": _get("phone", "telephone", "phone number"),
        "website": _get("website", "url"),
        "description": _get("description"),
        "categoryName": _get("category", "categoryName"),
        "location": {
            "lat": row.get("lat") or row.get("latitude"),
            "lng": row.get("lng") or row.get("longitude"),
        },
    }


def _save_results(results: List[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".json":
        with open(output_path, "w") as fp:
            json.dump(results, fp, indent=2)
    else:
        import pandas as pd  # late import to keep top clean

        pd.DataFrame(results).to_csv(output_path, index=False)


###############################################################################
# Main
###############################################################################


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate pharmacies with Perplexity (sonar/sonar-pro)")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--excel", help="Input Excel file path")
    input_group.add_argument(
        "--input-csv",
        help="Existing CSV of pharmacies to re-classify (e.g. sonar_pro_trial.csv)",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=0.2,
        help="Fraction of rows to sample (default 0.2 = 20%%)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of rows per API batch (default 10)",
    )
    parser.add_argument(
        "--model",
        choices=["sonar-pro", "sonar"],
        default="sonar-pro",
        help="Perplexity model to use (default: sonar-pro)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV/JSON path (default depends on model)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed for reproducible sampling (default 42)",
    )
    args = parser.parse_args()

    if args.excel:
        excel_path = Path(args.excel)
        if not excel_path.exists():
            sys.exit(f"Input Excel file not found: {excel_path}")

        df = pd.read_excel(excel_path, header=None)
        # If first row seems like headers (strings), promote it
        if all(isinstance(x, str) for x in df.iloc[0].tolist()):
            df.columns = df.iloc[0]
            df = df[1:]
        if df.empty:
            sys.exit("Spreadsheet is empty – nothing to process.")

        sample_df = df.sample(frac=args.sample_fraction, random_state=args.random_state)
        total = len(sample_df)
        print(
            f"Selected {total} rows (~{args.sample_fraction*100:.0f}% of {len(df)} rows) from {excel_path.name}"
        )
    else:
        csv_path = Path(args.input_csv)
        if not csv_path.exists():
            sys.exit(f"Input CSV file not found: {csv_path}")
        sample_df = pd.read_csv(csv_path)
        if sample_df.empty:
            sys.exit("CSV file is empty – nothing to process.")
        total = len(sample_df)
        print(f"Re-classifying {total} existing rows from {csv_path.name}")

    # Prepare Perplexity client – force_reclassification ensures fresh queries
    client = PerplexityClient(model_name=args.model, force_reclassification=True)

    results: List[Dict] = []

    for start in range(0, total, args.batch_size):
        batch = sample_df.iloc[start : start + args.batch_size]
        print(f"Processing rows {start + 1}-{min(start+args.batch_size, total)} / {total} …")
        for _, row in batch.iterrows():
            pharmacy_dict = _row_to_pharmacy(row)
            try:
                cls = client.classify_pharmacy(pharmacy_dict)
            except Exception as exc:  # noqa: BLE001
                print(f"❌ Classification failed for '{pharmacy_dict['name'][:50]}' – {exc}")
                cls = None
            # Ensure cls is a dict
            if not cls or not isinstance(cls, dict):
                cls = {
                    "classification": "error",
                    "confidence": 0.0,
                    "explanation": "No classification – parse failure or exception",
                }
            # Combine original row data with classification output
            combined = {**row.to_dict(), **cls}
            results.append(combined)

    default_out = f"results/{args.model.replace('-', '_')}_validation.csv"
    output_path = Path(args.output or default_out)
    _save_results(results, output_path)
    print(f"✅ Saved {len(results)} results to {output_path}")


if __name__ == "__main__":
    main()
