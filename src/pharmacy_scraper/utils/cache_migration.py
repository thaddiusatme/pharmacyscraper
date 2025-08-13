from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class MigrationStats:
    scanned: int = 0
    migrated: int = 0
    skipped_existing: int = 0
    errors: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "scanned": self.scanned,
            "migrated": self.migrated,
            "skipped_existing": self.skipped_existing,
            "errors": self.errors,
        }


def migrate_cache_keys(cache_dir: str | Path, business_type: str, *, dry_run: bool = False, overwrite: bool = False) -> MigrationStats:
    """Duplicate legacy untyped cache files to business_type-typed names.

    For each JSON file in cache_dir that does NOT already start with
    "{business_type}:" prefix, create a copy with that prefix. For example,
    "foo_bar.json" -> "vet_clinic:foo_bar.json".

    If the destination already exists and overwrite=False, skip it.

    Returns MigrationStats with counts.
    """
    stats = MigrationStats()
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    for p in cache_path.glob("*.json"):
        stats.scanned += 1
        name = p.name
        prefix = f"{business_type}:"
        if name.startswith(prefix):
            # Already typed for this business_type
            continue
        dest = cache_path / f"{prefix}{name}"
        if dest.exists() and not overwrite:
            stats.skipped_existing += 1
            continue
        try:
            if dry_run:
                stats.migrated += 1
            else:
                shutil.copyfile(p, dest)
                # sanity check: ensure JSON is valid at destination
                try:
                    json.loads(dest.read_text())
                except Exception:
                    # If JSON invalid, count as error but keep file for investigation
                    stats.errors += 1
                stats.migrated += 1
        except Exception:
            stats.errors += 1
    return stats


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Migrate cache keys to include business_type prefix")
    ap.add_argument("cache_dir", help="Path to cache directory containing *.json files")
    ap.add_argument("business_type", help="Business type prefix to apply (e.g., 'vet_clinic')")
    ap.add_argument("--dry-run", action="store_true", help="Do not write files; report what would change")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing typed files if present")
    return ap.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    stats = migrate_cache_keys(args.cache_dir, args.business_type, dry_run=args.dry_run, overwrite=args.overwrite)
    print(json.dumps({"business_type": args.business_type, **stats.to_dict()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
