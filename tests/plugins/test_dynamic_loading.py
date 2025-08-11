import pytest


# We'll load classes by import path using the format "module.sub:ClassName"
# Dummy plugin classes live in tests.plugins.dummies for TDD isolation.


def test_load_class_by_import_path_colon():
    from pharmacy_scraper.plugins.registry import load_class

    cls = load_class("tests.plugins.dummies:DummySource")
    assert cls.__name__ == "DummySource"

    cls2 = load_class("tests.plugins.dummies:DummyClassifier")
    assert cls2.__name__ == "DummyClassifier"


def test_load_class_invalid_path_raises():
    from pharmacy_scraper.plugins.registry import load_class

    with pytest.raises(ValueError):
        load_class("")
    with pytest.raises(ValueError):
        load_class("no_such_module:Nope")
    with pytest.raises(ValueError):
        load_class("tests.plugins.dummies:")


def test_registry_load_and_register():
    from pharmacy_scraper.plugins.registry import PluginRegistry
    from pharmacy_scraper.plugins.interfaces import DataSourcePlugin, ClassifierPlugin

    reg = PluginRegistry()
    reg.load_and_register([
        "tests.plugins.dummies:DummySource",
        "tests.plugins.dummies:DummyClassifier",
    ])

    srcs = reg.data_sources()
    clfs = reg.classifiers()

    assert any(issubclass(s, DataSourcePlugin) and s.__name__ == "DummySource" for s in srcs)
    assert any(issubclass(c, ClassifierPlugin) and c.__name__ == "DummyClassifier" for c in clfs)
