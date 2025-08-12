"""CSV Source Plugin

A simple data source plugin that reads pharmacy-like items from a CSV file.

Config keys (cfg):
- path (required): path to the CSV file
- encoding (optional): file encoding, default 'utf-8'
- delimiter (optional): CSV delimiter, default ','

Returns: list[dict]
"""
from __future__ import annotations

import csv
from typing import Any, Dict, List

from .interfaces import DataSourcePlugin


class CSVSourcePlugin(DataSourcePlugin):
    name = "csv_source"

    def fetch(self, query: Dict[str, Any], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not cfg or not cfg.get("path"):
            raise ValueError("CSVSourcePlugin requires 'path' in cfg")
        path = cfg["path"]
        encoding = cfg.get("encoding", "utf-8")
        delimiter = cfg.get("delimiter", ",")

        items: List[Dict[str, Any]] = []
        with open(path, "r", encoding=encoding, newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                items.append(dict(row))
        return items
