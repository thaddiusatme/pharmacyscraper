from __future__ import annotations

from typing import Type

from .interfaces import DataSourcePlugin, BaseClassifierPlugin


def _require_name_attr(cls: type) -> None:
    if not hasattr(cls, "name") or not isinstance(getattr(cls, "name"), str) or not getattr(cls, "name").strip():
        raise ValueError("Plugin class must define non-empty string attribute 'name'")


def validate_data_source_plugin(cls: Type[DataSourcePlugin]) -> None:
    if not isinstance(cls, type) or not issubclass(cls, DataSourcePlugin):
        raise ValueError("Plugin must subclass DataSourcePlugin")
    _require_name_attr(cls)
    if not hasattr(cls, "fetch") or not callable(getattr(cls, "fetch")):
        raise ValueError("DataSourcePlugin must implement a callable 'fetch' method")


def validate_classifier_plugin(cls: Type[BaseClassifierPlugin]) -> None:
    if not isinstance(cls, type) or not issubclass(cls, BaseClassifierPlugin):
        raise ValueError("Plugin must subclass BaseClassifierPlugin")
    _require_name_attr(cls)
    if not hasattr(cls, "classify") or not callable(getattr(cls, "classify")):
        raise ValueError("ClassifierPlugin must implement a callable 'classify' method")
