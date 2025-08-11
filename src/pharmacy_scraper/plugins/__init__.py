"""Plugin system for pharmacy_scraper.

This package defines interfaces (ABCs) for plugins and a registry for
runtime discovery and management of plugins.
"""

from .interfaces import DataSourcePlugin, ClassifierPlugin  # re-export
from .registry import PluginRegistry  # re-export

__all__ = [
    "DataSourcePlugin",
    "ClassifierPlugin",
    "PluginRegistry",
]
