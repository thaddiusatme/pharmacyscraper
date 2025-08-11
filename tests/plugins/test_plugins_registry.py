import types
import pytest

from typing import List

# Tests assume the plugin interfaces and registry live under
# pharmacy_scraper.plugins.{interfaces,registry}


def test_interfaces_exist_and_enforce_required_methods():
    from pharmacy_scraper.plugins.interfaces import DataSourcePlugin, ClassifierPlugin

    # Good: concrete subclasses implementing abstract methods should be instantiable
    class MySource(DataSourcePlugin):
        name = "my_source"
        def fetch(self, query: dict, cfg: dict) -> list:
            return []

    class MyClassifier(ClassifierPlugin):
        name = "my_classifier"
        def classify(self, item: dict, cfg: dict) -> dict:
            return {"label": "unknown", "score": 0.0}

    MySource()  # should not raise
    MyClassifier()  # should not raise

    # Bad: missing required method should raise TypeError on instantiation
    class BadSource(DataSourcePlugin):
        name = "bad_source"
        # missing fetch
        pass

    with pytest.raises(TypeError):
        BadSource()


def test_registry_register_and_list_plugins_by_kind():
    from pharmacy_scraper.plugins.interfaces import DataSourcePlugin, ClassifierPlugin
    from pharmacy_scraper.plugins.registry import PluginRegistry

    registry = PluginRegistry()

    class SrcA(DataSourcePlugin):
        name = "src_a"
        def fetch(self, query: dict, cfg: dict) -> list:
            return [1]

    class SrcB(DataSourcePlugin):
        name = "src_b"
        def fetch(self, query: dict, cfg: dict) -> list:
            return [2]

    class ClfA(ClassifierPlugin):
        name = "clf_a"
        def classify(self, item: dict, cfg: dict) -> dict:
            return {"label": "x", "score": 0.9}

    registry.register(SrcA)
    registry.register(SrcB)
    registry.register(ClfA)

    sources = registry.data_sources()
    classifiers = registry.classifiers()

    assert [cls.name for cls in sources] == ["src_a", "src_b"]
    assert [cls.name for cls in classifiers] == ["clf_a"]


def test_registry_rejects_non_plugin_types():
    from pharmacy_scraper.plugins.registry import PluginRegistry
    from pharmacy_scraper.plugins.interfaces import DataSourcePlugin

    registry = PluginRegistry()

    class NotAPlugin:
        pass

    with pytest.raises(ValueError):
        registry.register(NotAPlugin)  # type: ignore[arg-type]

    # also reject subclasses of wrong base when specifying expected kind
    class SomeSource(DataSourcePlugin):
        name = "ok"
        def fetch(self, query: dict, cfg: dict) -> list:
            return []

    # misusing internal API to show kind filtering works
    with pytest.raises(ValueError):
        registry._ensure_valid(SomeSource, expected_kind="classifier")
