import pytest


def test_registry_build_from_config_happy_path():
    from pharmacy_scraper.plugins.registry import PluginRegistry
    cfg = {
        "sources": ["tests.plugins.dummies:DummySource"],
        "classifiers": ["tests.plugins.dummies:DummyClassifier"],
    }
    reg = PluginRegistry()
    reg.build_from_config(cfg)

    assert [c.__name__ for c in reg.data_sources()] == ["DummySource"]
    assert [c.__name__ for c in reg.classifiers()] == ["DummyClassifier"]


def test_registry_build_from_config_errors():
    from pharmacy_scraper.plugins.registry import PluginRegistry

    # missing keys should default to empty and not raise
    reg = PluginRegistry()
    reg.build_from_config({})
    assert reg.data_sources() == []
    assert reg.classifiers() == []

    # bad class path raises ValueError with useful message
    reg2 = PluginRegistry()
    with pytest.raises(ValueError):
        reg2.build_from_config({"sources": ["no.module:Nope"]})
