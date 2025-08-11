"""Adapters to expose existing components as plugins.

- ApifySourceAdapter wraps ApifyCollector as a DataSourcePlugin
- ClassifierAdapter wraps classification.Classifier as a ClassifierPlugin
"""
from __future__ import annotations

from typing import Any, Dict, List

from .interfaces import DataSourcePlugin, ClassifierPlugin


class ApifySourceAdapter(DataSourcePlugin):
    name = "apify_source"

    def fetch(self, query: dict, cfg: dict) -> list:
        # Lazy import to keep adapter lightweight and easy to test/patch
        from pharmacy_scraper.api.apify_collector import ApifyCollector

        # Instantiate collector; rely on env or provided config for auth
        collector = ApifyCollector()

        # Shape of config is flexible in collector; pass through cfg or
        # embed the query inside to preserve information for future use.
        config = dict(cfg or {})
        if query:
            # Keep a conventional key for query info
            config.setdefault("query", query)

        results = collector.collect_pharmacies(config)
        # Ensure list output
        return list(results or [])


class ClassifierAdapter(ClassifierPlugin):
    name = "default_classifier"

    def classify(self, item: dict, cfg: dict) -> dict:
        from pharmacy_scraper.classification.classifier import Classifier

        use_llm = True if cfg is None else bool(cfg.get("use_llm", True))
        clf = Classifier()
        result = clf.classify(item, use_llm=use_llm)

        # Map ClassificationResult-like object to a generic dict contract
        # Expect attributes: is_chain, is_compounding, confidence
        label: str
        if getattr(result, "is_chain", False):
            label = "chain"
        elif getattr(result, "is_compounding", False):
            label = "compounding"
        else:
            label = "independent"

        score = float(getattr(result, "confidence", 0.0) or 0.0)
        return {"label": label, "score": score}
