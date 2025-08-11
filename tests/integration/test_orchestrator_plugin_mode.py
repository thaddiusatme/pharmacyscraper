import json
import types
from pathlib import Path


def test_orchestrator_runs_with_plugins_dummy_only(tmp_path, monkeypatch):
    # Prepare config file enabling plugin mode with dummy plugins
    config = {
        "plugin_mode": True,
        "plugins": {
            "sources": ["tests.plugins.dummies:DummySource"],
            "classifiers": ["tests.plugins.dummies:DummyClassifier"],
        },
        "plugin_config": {
            "DummySource": {"max_results": 3},
            "DummyClassifier": {"use_llm": False},
        },
        # minimal required legacy fields
        "api_keys": {},
        "output_dir": str(tmp_path / "out"),
        "cache_dir": str(tmp_path / "cache"),
        "locations": [],
    }
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(config))

    from pharmacy_scraper.orchestrator.pipeline_orchestrator import PipelineOrchestrator

    orch = PipelineOrchestrator(str(cfg_path))
    out_path = orch.run()

    assert isinstance(out_path, Path)
    assert out_path.exists()


def test_orchestrator_runs_with_adapters_monkeypatched(tmp_path, monkeypatch):
    # Monkeypatch ApifyCollector and Classifier used by adapters
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
            is_chain = "cvs" in (pharmacy.get("name", "").lower())
            return types.SimpleNamespace(
                is_chain=is_chain, is_compounding=False, confidence=0.9
            )

    monkeypatch.setattr(apify_mod, "ApifyCollector", FakeCollector)
    monkeypatch.setattr(clf_mod, "Classifier", FakeClassifier)

    config = {
        "plugin_mode": True,
        "plugins": {
            "sources": ["pharmacy_scraper.plugins.adapters:ApifySourceAdapter"],
            "classifiers": ["pharmacy_scraper.plugins.adapters:ClassifierAdapter"],
        },
        "plugin_config": {
            "ApifySourceAdapter": {"max_results": 3},
            "ClassifierAdapter": {"use_llm": False},
        },
        "api_keys": {},
        "output_dir": str(tmp_path / "out"),
        "cache_dir": str(tmp_path / "cache"),
        "locations": [],
    }
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(config))

    from pharmacy_scraper.orchestrator.pipeline_orchestrator import PipelineOrchestrator

    orch = PipelineOrchestrator(str(cfg_path))
    out_path = orch.run()

    assert isinstance(out_path, Path)
    assert out_path.exists()
    # read results and ensure two records
    data = json.loads(out_path.read_text())
    assert isinstance(data, list)
    assert len(data) == 2
