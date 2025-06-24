#!/usr/bin/env python3
"""limit_independents_per_state.py

Utility script that post-processes a classified spreadsheet to:
1. Keep **at most N independent pharmacies per US state** (default 25).
2. Output a second Excel sheet listing states that ended up with < N independents.

Example:
    python scripts/limit_independents_per_state.py \
        --input reclass_post_mt_filtered.xlsx \
        --output reclass_post_mt_capped.xlsx \
        --limit 25
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

###############################################################################
# Helper
###############################################################################


def _norm_cols(df: pd.DataFrame) -> dict[str, str]:
    """Map lowercase column names -> original names for flexible access."""
    return {str(c).lower(): c for c in df.columns}


###############################################################################
# Main
###############################################################################


def main() -> None:  # noqa: D401 – simple CLI
    parser = argparse.ArgumentParser(
        description="Cap independent pharmacies per state and report shortages"
    )
    parser.add_argument("--input", required=True, help="Input classified Excel file")
    parser.add_argument(
        "--output",
        default=None,
        help="Output Excel filename (defaults to *_capped.xlsx)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Maximum independent pharmacies to keep per state (default 25)",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"❌  Input file not found: {in_path}")

    df = pd.read_excel(in_path)
    cols = _norm_cols(df)

    # Detect columns
    state_col = cols.get("state")
    is_chain_col = cols.get("is_chain")

    if state_col is None or is_chain_col is None:
        raise SystemExit("❌  Could not find 'state' or 'is_chain' columns in sheet")

    # Treat NaN as False for is_chain
    df[is_chain_col] = df[is_chain_col].fillna(False).astype(bool)

    # Independent rows classified by LLM only
    reason_col = cols.get("reason")
    if reason_col is None:
        raise SystemExit("❌  'reason' column missing – cannot filter LLM results")

    independents = df[
        (~df[is_chain_col])
        & df[reason_col].fillna("").str.startswith("LLM classification: independent")
    ]

    # Keep at most N per state
    capped: List[pd.DataFrame] = []
    shortage_states: dict[str, int] = {}

    for state, group in independents.groupby(state_col):
        capped.append(group.head(args.limit))
        if len(group) < args.limit:
            shortage_states[state] = len(group)

    capped_df = pd.concat(capped, ignore_index=True)

    # Save capped independents sheet
    out_path = Path(args.output or f"{in_path.stem}_capped{in_path.suffix}")
    capped_df.to_excel(out_path, index=False)
    print(f"✅  Saved capped file to {out_path} (≤{args.limit} independents/state)")

    # Build shortage report DataFrame
    if shortage_states:
        report_df = (
            pd.Series(shortage_states, name="count")
            .rename_axis("state")
            .reset_index()
            .sort_values("state")
        )
        report_path = out_path.with_name(f"{out_path.stem}_shortages{out_path.suffix}")
        report_df.to_excel(report_path, index=False)
        print(
            f"⚠️  {len(shortage_states)} states have fewer than {args.limit} independents. "
            f"Report saved to {report_path}"
        )
    else:
        print(f"🎉 All states have at least {args.limit} independents.")


if __name__ == "__main__":
    main()
