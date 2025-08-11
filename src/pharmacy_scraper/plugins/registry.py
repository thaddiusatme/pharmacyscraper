"""Simple in-process plugin registry.

Supports registering plugins that implement the ABCs defined in
`pharmacy_scraper.plugins.interfaces` and listing them by kind.
"""
from __future__ import annotations

import importlib
from typing import List, Sequence, Type, Union

from .interfaces import DataSourcePlugin, ClassifierPlugin, BasePlugin

PluginType = Union[Type[DataSourcePlugin], Type[ClassifierPlugin]]


def load_class(path: str) -> type:
    """Load a class from an import path like 'module.sub:ClassName'.

    Raises ValueError on invalid input or import problems.
    """
    if not path or ":" not in path:
        raise ValueError("Import path must be in format 'module:Class'")
    module_name, class_name = path.split(":", 1)
    if not module_name or not class_name:
        raise ValueError("Import path must specify both module and class")
    try:
        mod = importlib.import_module(module_name)
        cls = getattr(mod, class_name)
    except Exception as e:
        raise ValueError(f"Failed to load '{path}': {e}") from e
    if not isinstance(cls, type):
        raise ValueError(f"Target '{path}' is not a class")
    return cls


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

    def load_and_register(self, paths: Sequence[str]) -> None:
        """Load classes by import path and register them by kind."""
        for p in paths:
            cls = load_class(p)
            self.register(cls)  # _ensure_valid inside register

    def data_sources(self) -> List[Type[DataSourcePlugin]]:
        return list(self._sources)

    def classifiers(self) -> List[Type[ClassifierPlugin]]:
        return list(self._classifiers)
