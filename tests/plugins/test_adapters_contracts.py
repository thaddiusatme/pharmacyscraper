import types


def test_apify_source_adapter_fetch_uses_collector(monkeypatch):
    # Arrange: monkeypatch ApifyCollector to avoid real API
    from pharmacy_scraper.plugins.registry import load_class

    calls = {}

    class FakeCollector:
        def __init__(self, *a, **kw):
            calls['init'] = True
        def collect_pharmacies(self, config):
            calls['collect_cfg'] = config
            return [
                {"name": "Indie Pharmacy", "address": "123 St"},
                {"name": "CVS Pharmacy", "address": "456 Ave"},
            ]

    import pharmacy_scraper.api.apify_collector as apify_mod
    monkeypatch.setattr(apify_mod, 'ApifyCollector', FakeCollector)

    # Act
    Adapter = load_class("pharmacy_scraper.plugins.adapters:ApifySourceAdapter")
    src = Adapter()
    out = src.fetch(query={"query": "pharmacy seattle"}, cfg={"max_results": 10})

    # Assert
    assert calls.get('init') is True
    assert isinstance(out, list) and len(out) == 2
    assert any("Indie" in (out[0].get("name") or "") for _ in [0])


def test_classifier_adapter_classify_wraps_classifier(monkeypatch):
    # Arrange: monkeypatch Classifier.classify to deterministic output
    from pharmacy_scraper.plugins.registry import load_class
    import pharmacy_scraper.classification.classifier as clf_mod

    class FakeClassifier:
        def __init__(self, *a, **kw):
            pass
        def classify(self, pharmacy, use_llm=True):
            return types.SimpleNamespace(is_chain=False, is_compounding=True, confidence=0.9)

    monkeypatch.setattr(clf_mod, 'Classifier', FakeClassifier)

    Adapter = load_class("pharmacy_scraper.plugins.adapters:ClassifierAdapter")
    clf = Adapter()
    result = clf.classify({"name": "Compounding Rx"}, cfg={"use_llm": False})

    assert isinstance(result, dict)
    assert result["label"] in ("compounding", "unknown", "independent", "chain")
    assert 0.0 <= result.get("score", 0.0) <= 1.0
