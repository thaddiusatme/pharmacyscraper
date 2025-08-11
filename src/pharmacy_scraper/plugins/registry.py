"""Simple in-process plugin registry.

Supports registering plugins that implement the ABCs defined in
`pharmacy_scraper.plugins.interfaces` and listing them by kind.
"""
from __future__ import annotations

from typing import List, Sequence, Type, Union

from .interfaces import DataSourcePlugin, ClassifierPlugin, BasePlugin

PluginType = Union[Type[DataSourcePlugin], Type[ClassifierPlugin]]


class PluginRegistry:
    """Holds known plugins and can filter by kind."""

    def __init__(self) -> None:
        self._sources: list[Type[DataSourcePlugin]] = []
        self._classifiers: list[Type[ClassifierPlugin]] = []

    def _ensure_valid(self, cls: type, expected_kind: str | None = None) -> None:
        """Validate plugin class type and optional expected kind.

        Raises ValueError on invalid class.
        """
        if not isinstance(cls, type):
            raise ValueError("Plugin must be a class")

        if issubclass(cls, DataSourcePlugin):
            kind = "source"
        elif issubclass(cls, ClassifierPlugin):
            kind = "classifier"
        else:
            raise ValueError("Plugin must subclass a supported plugin base class")

        if expected_kind and expected_kind != kind:
            raise ValueError(f"Expected plugin kind '{expected_kind}', got '{kind}'")

    def register(self, cls: PluginType) -> None:
        """Register a plugin class by its kind."""
        self._ensure_valid(cls)
        if issubclass(cls, DataSourcePlugin):
            self._sources.append(cls)
        elif issubclass(cls, ClassifierPlugin):
            self._classifiers.append(cls)

    def data_sources(self) -> List[Type[DataSourcePlugin]]:
        return list(self._sources)

    def classifiers(self) -> List[Type[ClassifierPlugin]]:
        return list(self._classifiers)
