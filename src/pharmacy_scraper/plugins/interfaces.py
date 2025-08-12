"""Plugin interfaces (ABCs) for the pharmacy scraper.

These define the contracts that plugins must implement. We keep them very
small/minimal to make it easy to implement new plugins and to support TDD.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BasePlugin(ABC):
    """Base class for all plugins.

    Attributes
    -----------
    name: str
        A unique, short identifier for the plugin. Used for discovery and
        per-plugin configuration selection.
    """

    name: str


class DataSourcePlugin(BasePlugin):
    """Interface for data source plugins that fetch pharmacy candidates."""

    @abstractmethod
    def fetch(self, query: dict, cfg: dict) -> list:
        """Fetch entities for a query using provided config.

        Parameters
        ----------
        query: dict
            Query object/params (kept generic for now).
        cfg: dict
            Configuration specific to this plugin.
        Returns
        -------
        list
            A list of items (e.g., pharmacies) found.
        """


class BaseClassifierPlugin(BasePlugin, ABC):
    """Base interface for classification plugins with optional hooks.

    Plugins may override the lightweight detection hooks to provide
    fast-path decisions without invoking heavier logic.
    """

    def detect_chain(self, item: dict, cfg: dict) -> bool | None:  # optional
        """Optional fast-path check for chain detection.

        Return True/False to provide an early determination, or None to defer
        to full classification.
        """
        return None

    def detect_compounding(self, item: dict, cfg: dict) -> bool | None:  # optional
        """Optional fast-path check for compounding detection.

        Return True/False to provide an early determination, or None to defer
        to full classification.
        """
        return None

    @abstractmethod
    def classify(self, item: dict, cfg: dict) -> dict:
        """Classify a single item.

        Returns a mapping with at least keys like {"label": str, "score": float}.
        """


# Backwards-compatible alias used across the codebase/registry
class ClassifierPlugin(BaseClassifierPlugin):
    pass
