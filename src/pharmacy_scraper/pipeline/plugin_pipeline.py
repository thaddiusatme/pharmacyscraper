"""Minimal plugin-driven pipeline composition.

This runner builds a registry from config, instantiates sources and classifiers,
fetches items, classifies them, and returns labeled results.
"""
from __future__ import annotations

from typing import Any, Dict, List

from pharmacy_scraper.plugins.registry import PluginRegistry


def _per_plugin_cfg(plugin_cls: type, all_cfg: Dict[str, Dict]) -> Dict:
    # Look up per-plugin config by class name, fallback to empty
    return dict(all_cfg.get(getattr(plugin_cls, "__name__", ""), {}))


def run_pipeline(config: Dict[str, Any], query: Dict | None = None) -> List[Dict]:
    plugins_cfg = (config or {}).get("plugins", {})
    per_cfg = (config or {}).get("plugin_config", {})

    # Build registry
    registry = PluginRegistry()
    registry.build_from_config(plugins_cfg)

    results: List[Dict] = []

    # Instantiate sources
    sources = [cls() for cls in registry.data_sources()]
    classifiers = [cls() for cls in registry.classifiers()]

    # Keep it simple: use first classifier if present
    classifier = classifiers[0] if classifiers else None

    for src_cls, src in zip(registry.data_sources(), sources):
        src_cfg = _per_plugin_cfg(src_cls, per_cfg)
        items = src.fetch(query or {}, src_cfg)
        for it in items:
            if classifier:
                clf_cfg = _per_plugin_cfg(type(classifier), per_cfg)
                out = classifier.classify(it, clf_cfg)
                # Ensure dict output and merge with original item to preserve fields
                if isinstance(out, dict):
                    base = it if isinstance(it, dict) else {}
                    merged = {**base, **out}
                    results.append(merged)
                else:
                    base = it if isinstance(it, dict) else {}
                    results.append({**base, "label": "unknown", "score": 0.0})
            else:
                # No classifier – pass through with default label
                base = it if isinstance(it, dict) else {}
                results.append({**base, "label": "unknown", "score": 0.0})

    return results
