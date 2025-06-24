#!/usr/bin/env python3
"""quick_reclass_post_mt.py

Batch re-classification helper.

Loads the original spreadsheet that was manually classified up to Montana ("MT")
and automatically classifies the remaining rows using the hybrid rule-based +
Perplexity workflow from `src.classification`.

Example:
    python scripts/quick_reclass_post_mt.py \
        --excel "Copy of Untitled spreadsheet.xlsx" \
        --output reclassified.xlsx \
        --batch-size 20 \
        --model sonar-pro

Environment:
    Requires PERPLEXITY_API_KEY for live LLM calls.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Add repository root for imports when executed directly
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.classification.classifier import Classifier  # noqa: E402
from src.classification.perplexity_client import PerplexityClient  # noqa: E402

###############################################################################
# Helper functions
###############################################################################


def _fix_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Clean up DataFrame headers for common malformed cases.

    If the first column header appears to be purely numeric (e.g. "2") or an
    automatically generated placeholder like "Unnamed: 0", rename it to
    "pharmacy name" so downstream name detection works.
    """
    new_cols = list(df.columns)

    # Detect if we already have a usable name column
    lowered = {str(c).lower() for c in new_cols}
    name_variants = {
        "name",
        "pharmacy name",
        "pharmacy",
        "business name",
        "store name",
    }
    if lowered & name_variants:
        return df  # Nothing to do

    first_col = str(new_cols[0])
    if first_col.lower().startswith("unnamed") or first_col.isdigit():
        new_cols[0] = "pharmacy name"
        df.columns = new_cols
    return df


def _norm_cols(df: pd.DataFrame) -> Dict[str, str]:
    """Return a dict mapping lowercase column names -> actual column names."""
    return {str(c).lower(): c for c in df.columns}


def _detect_state_col(df: pd.DataFrame) -> str | None:
    """Heuristically find the column that holds US state code values."""
    for candidate in ("state", "st", "province"):
        if candidate in _norm_cols(df):
            return _norm_cols(df)[candidate]
    return None


def _row_to_pharmacy(row: pd.Series) -> Dict:
    """Convert a DataFrame row to the dict format expected by Classifier."""
    cols = _norm_cols(row.to_frame().T)

    def _get(*keys: str, default: str = "") -> str:
        for k in keys:
            if k.lower() in cols:
                return str(row[cols[k.lower()]]).strip()
        return default

    return {
        "name": _get(
            "name",
            "title",
            "pharmacy name",
            "pharmacy",
            "business name",
            "store name",
        ),
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

###############################################################################
# Main entry point
###############################################################################


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-classify spreadsheet rows with Perplexity")
    parser.add_argument("--excel", required=True, help="Input Excel filename")
    parser.add_argument("--output", default=None, help="Output filename (defaults to *_reclass.xlsx)")
    parser.add_argument("--batch-size", type=int, default=20, help="Rows per API batch (default 20)")
    parser.add_argument("--model", choices=["sonar-pro", "sonar"], default="sonar", help="Perplexity model to use (default sonar for lower cost)")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Process at most this many rows (useful for a quick sample run)",
    )
    parser.add_argument(
        "--only-new",
        action="store_true",
        help="If set, the output file will contain only the newly classified rows instead of the full sheet",
    )
    parser.add_argument(
        "--after-state",
        default="MT",
        help="Skip rows up to and including this 2-letter state code (default MT)",
    )
    args = parser.parse_args()

    in_path = Path(args.excel)
    if not in_path.exists():
        sys.exit(f"❌  Input file not found: {in_path}")

    # Load spreadsheet (let pandas infer engine). We'll attempt to detect if the
    # first row is actually a header row by looking for common header keywords
    # such as "address", "city", or "state". This is more reliable than simply
    # checking that all values are strings (since real data rows are often all
    # strings too).
    df = pd.read_excel(in_path, header=None)

    COMMON_HEADER_TOKENS = {"address", "city", "state", "zip", "phone", "website"}
    if df.shape[0] > 0:
        first_row_vals = {str(x).strip().lower() for x in df.iloc[0].tolist() if pd.notna(x)}
        if first_row_vals & COMMON_HEADER_TOKENS:
            # Promote first row to header
            df.columns = df.iloc[0]
            df = df[1:]

    df = df.reset_index(drop=True)

    # Clean up problematic headers (e.g., first column numeric) so name detection works
    df = _fix_headers(df)

    state_col = _detect_state_col(df)

    # Determine which rows still need classification
    mask_to_classify = pd.Series(True, index=df.index)

    if state_col is not None:
        last_idx = df[df[state_col].astype(str).str.upper() == args.after_state.upper()].index.max()
        if pd.notna(last_idx):
            mask_to_classify &= df.index > last_idx
            print(f"⚙️  Skipping rows up to index {last_idx} (state {args.after_state.upper()}).")

    # If sheet already contains a 'classification' column, skip rows with a value
    if "classification" in _norm_cols(df):
        cls_col = _norm_cols(df)["classification"]
        mask_to_classify &= df[cls_col].isna() | (df[cls_col] == "")

    rows_to_process = df[mask_to_classify]

    # Optional sampling for quick test runs
    if args.max_rows is not None:
        rows_to_process = rows_to_process.iloc[: args.max_rows]

    total = len(rows_to_process)
    if total == 0:
        print("✅  All rows already classified – nothing to do.")
        return

    print(f"🔎  Will classify {total} rows using model '{args.model}' (batch={args.batch_size}) …")

    # Prepare classifier – PerplexityClient does actual API calls
    client = PerplexityClient(model_name=args.model)
    classifier = Classifier(client)

    classifications: List[Dict] = []
    for start in range(0, total, args.batch_size):
        end = min(start + args.batch_size, total)
        batch = rows_to_process.iloc[start:end]
        print(f" • Rows {start + 1}-{end} / {total}")
        pharmacies = [_row_to_pharmacy(r) for _, r in batch.iterrows()]
        batch_results = [classifier.classify_pharmacy(p) for p in pharmacies]
        classifications.extend(batch_results)

    # Merge back into DataFrame (index aligned)
    results_df = pd.DataFrame(classifications, index=rows_to_process.index)
    combined_df = df.join(results_df, how="left")

    if args.only_new:
        output_df = combined_df.loc[rows_to_process.index]
    else:
        output_df = combined_df

    out_path = Path(args.output or f"{in_path.stem}_reclass{in_path.suffix}")
    output_df.to_excel(out_path, index=False)
    print(f"✅  Saved updated spreadsheet to {out_path} ({'only new rows' if args.only_new else 'full sheet'})")


if __name__ == "__main__":
    main()
