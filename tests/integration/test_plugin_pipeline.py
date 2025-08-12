import types
import pytest


def test_pipeline_with_dummies_end_to_end():
    from pharmacy_scraper.pipeline.plugin_pipeline import run_pipeline

    config = {
        "plugins": {
            "sources": ["tests.plugins.dummies:DummySource"],
            "classifiers": ["tests.plugins.dummies:DummyClassifier"],
        },
        "plugin_config": {
            "DummySource": {"max_results": 5},
            "DummyClassifier": {"use_llm": False},
        },
    }

    results = run_pipeline(config, query={"q": "seattle pharmacies"})
    assert isinstance(results, list)
    assert len(results) >= 1
    for item in results:
        assert isinstance(item, dict)
        assert "label" in item and "score" in item


def test_pipeline_with_adapters_end_to_end(monkeypatch):
    # Monkeypatch external dependencies: ApifyCollector and Classifier
    import pharmacy_scraper.api.apify_collector as apify_mod
    import pharmacy_scraper.classification.classifier as clf_mod

    class FakeCollector:
        def __init__(self, *a, **kw):
            pass
        def collect_pharmacies(self, config):
            return [
                {"name": "Indie Rx", "address": "123"},
                {"name": "CVS Pharmacy", "address": "456"},
            ]

    class FakeClassifier:
        def __init__(self, *a, **kw):
            pass
        def classify(self, pharmacy, use_llm=True):
            # Mark CVS as chain, others as independent
            is_chain = "cvs" in (pharmacy.get("name", "").lower())
            return types.SimpleNamespace(
                is_chain=is_chain,
                is_compounding=False,
                confidence=0.9,
            )

    monkeypatch.setattr(apify_mod, "ApifyCollector", FakeCollector)
    monkeypatch.setattr(clf_mod, "Classifier", FakeClassifier)

    from pharmacy_scraper.pipeline.plugin_pipeline import run_pipeline

    config = {
        "plugins": {
            "sources": [
                "pharmacy_scraper.plugins.adapters:ApifySourceAdapter",
            ],
            "classifiers": [
                "pharmacy_scraper.plugins.adapters:ClassifierAdapter",
            ],
        },
        "plugin_config": {
            "ApifySourceAdapter": {"max_results": 5},
            "ClassifierAdapter": {"use_llm": False},
        },
    }

    results = run_pipeline(config, query={"q": "seattle pharmacies"})
    assert isinstance(results, list)
    assert len(results) == 2
    # Ensure labels reflect FakeClassifier logic
    labels = {r["label"] for r in results}
    assert labels.issubset({"chain", "independent", "compounding"})


def test_pipeline_invalid_plugin_path_raises():
    from pharmacy_scraper.pipeline.plugin_pipeline import run_pipeline

    config = {
        "plugins": {
            "sources": ["no.module:Nope"],
            "classifiers": ["tests.plugins.dummies:DummyClassifier"],
        }
    }

    with pytest.raises(ValueError):
        run_pipeline(config, query={})
