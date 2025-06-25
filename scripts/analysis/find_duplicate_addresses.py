#!/usr/bin/env python3
"""find_duplicate_addresses.py

Scan an Excel sheet and flag obviously duplicate addresses. Duplicates are
identified by a simple canonicalisation (lower-case, trim, collapse whitespace,
remove punctuation). Optionally, results can be written to a new Excel file
with a `dup_group` column plus a separate sheet listing only duplicates.

Usage:
    python scripts/find_duplicate_addresses.py \
        --input uo_to_mt.xlsx \
        --output uo_to_mt_dedupe_report.xlsx
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict

import pandas as pd

PUNCT_RE = re.compile(r"[.,;:/\\-]")


def canonical(addr: str) -> str:
    """Simple normalisation for address strings."""
    if not isinstance(addr, str):
        return ""
    addr = addr.lower()
    addr = PUNCT_RE.sub(" ", addr)  # remove common punctuation
    addr = re.sub(r"\s+", " ", addr)  # collapse whitespace
    return addr.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect duplicate addresses in Excel sheet")
    parser.add_argument("--input", required=True, help="Input Excel filename")
    parser.add_argument("--output", default=None, help="Output Excel filename (adds dup info)")
    parser.add_argument(
        "--address-col",
        default="Address",
        help="Column name that contains the address (case insensitive)",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"❌  File not found: {in_path}")

    df = pd.read_excel(in_path)
    # find address column (case-insensitive)
    addr_col_map: Dict[str, str] = {str(c).lower(): c for c in df.columns}
    if args.address_col.lower() not in addr_col_map:
        raise SystemExit(f"❌  Address column '{args.address_col}' not found in sheet")
    addr_col = addr_col_map[args.address_col.lower()]

    # compute canonical addresses
    df["_canon_addr"] = df[addr_col].astype(str).map(canonical)

    # group and mark duplicates
    dup_groups = df.groupby("_canon_addr").filter(lambda g: len(g) > 1)
    if dup_groups.empty:
        print("🎉  No duplicate addresses detected.")
    else:
        print(f"⚠️  Found {dup_groups.shape[0]} rows across {dup_groups['_canon_addr'].nunique()} duplicate groups.")

    # assign group id
    canon_to_id = {canon: i + 1 for i, canon in enumerate(dup_groups["_canon_addr"].unique())}
    df["dup_group"] = df["_canon_addr"].map(canon_to_id).fillna(0).astype(int)

    # write output
    out_path = Path(args.output or f"{in_path.stem}_dedupe_report{in_path.suffix}")
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.drop(columns=["_canon_addr"]).to_excel(writer, index=False, sheet_name="all_rows")
        if not dup_groups.empty:
            dup_groups.drop(columns=["_canon_addr"]).to_excel(writer, index=False, sheet_name="duplicates")
    print(f"✅  Report written to {out_path}")


if __name__ == "__main__":
    main()
