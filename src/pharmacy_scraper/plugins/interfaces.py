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


class ClassifierPlugin(BasePlugin):
    """Interface for classification plugins."""

    @abstractmethod
    def classify(self, item: dict, cfg: dict) -> dict:
        """Classify a single item.

        Returns a mapping with at least keys like {"label": str, "score": float}.
        """
