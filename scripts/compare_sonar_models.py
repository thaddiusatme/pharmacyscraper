#!/usr/bin/env python3
"""compare_sonar_models.py

Compare classification outputs between two Perplexity runs (e.g. *sonar* vs
*sonar-pro*).

It produces:
• Console summary with total rows, agreement count, disagreement count, and
  simple confusion matrix (counts of each classification pair).
• Optional CSV of differing rows for manual inspection.

Usage example:
    python scripts/compare_sonar_models.py \
        --pro results/sonar_pro_trial.csv \
        --vanilla results/sonar_validation.csv \
        --output results/sonar_diff.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path} is empty")
    return df


def _make_key(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    return df[cols].astype(str).agg("|".join, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare sonar vs sonar-pro outputs")
    parser.add_argument("--pro", required=True, help="CSV file from sonar-pro run")
    parser.add_argument("--vanilla", required=True, help="CSV file from sonar run")
    parser.add_argument(
        "--output",
        help="Optional CSV path to save differing rows (default: results/sonar_differences.csv)",
    )
    parser.add_argument(
        "--key-cols",
        nargs="*",
        default=["name", "address"],
        help="Columns used to align rows between the two CSVs (default: name address)",
    )
    args = parser.parse_args()

    pro_df = _load_csv(Path(args.pro))
    vanilla_df = _load_csv(Path(args.vanilla))

    key_cols = args.key_cols
    for col in key_cols:
        if col not in pro_df.columns or col not in vanilla_df.columns:
            raise KeyError(f"Key column '{col}' missing in one of the CSVs")

    pro_df["_key"] = _make_key(pro_df, key_cols)
    vanilla_df["_key"] = _make_key(vanilla_df, key_cols)

    merged = pro_df.merge(
        vanilla_df,
        on="_key",
        suffixes=("_pro", "_vanilla"),
        how="inner",
    )

    total = len(merged)
    if total == 0:
        raise ValueError("No overlapping rows based on key columns")

    agreement_mask = (
        merged["classification_pro"].fillna("NA")
        == merged["classification_vanilla"].fillna("NA")
    )

    agree = agreement_mask.sum()
    disagree = total - agree

    print("=== Comparison Summary ===")
    print(f"Total overlapping rows: {total}")
    print(f"Agreement: {agree} ({agree/total:.1%})")
    print(f"Disagreement: {disagree} ({disagree/total:.1%})\n")

    # Confusion matrix counts
    conf = (
        merged.groupby(["classification_pro", "classification_vanilla"])  # type: ignore[arg-type]
        .size()
        .unstack(fill_value=0)
    )
    print("Confusion matrix (pro rows vs vanilla columns):")
    print(conf.to_string())

    if disagree > 0:
        diff_rows = merged[~agreement_mask].copy()
        # Move key columns to front for readability
        # Resolve potential suffixes for key columns after merge
        resolved_keys = []
        for col in key_cols:
            if col in diff_rows.columns:
                resolved_keys.append(col)
            elif f"{col}_pro" in diff_rows.columns:
                resolved_keys.append(f"{col}_pro")
            elif f"{col}_vanilla" in diff_rows.columns:
                resolved_keys.append(f"{col}_vanilla")
        front_cols = resolved_keys + [c for c in diff_rows.columns if c not in resolved_keys]
        diff_rows = diff_rows[front_cols]

        out_path = Path(args.output or "results/sonar_differences.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        diff_rows.to_csv(out_path, index=False)
        print(f"\nSaved {disagree} differing rows to {out_path}")


if __name__ == "__main__":
    main()
