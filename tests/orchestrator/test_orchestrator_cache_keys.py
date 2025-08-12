import json
from pathlib import Path

import pytest

from pharmacy_scraper.orchestrator.pipeline_orchestrator import PipelineOrchestrator


def _write_cfg(tmp_path: Path, *, business_type: str, cache_dir: Path, output_dir: Path) -> Path:
    cfg = {
        "api_keys": {},
        "output_dir": str(output_dir),
        "cache_dir": str(cache_dir),
        "locations": [
            {"state": "CA", "cities": ["San Francisco"], "queries": ["independent pharmacy"]}
        ],
        "business_type": business_type,
    }
    cfg_path = tmp_path / f"config_{business_type}.json"
    cfg_path.write_text(json.dumps(cfg))
    return cfg_path


@pytest.fixture()
def tmp_dirs(tmp_path):
    cache_dir = tmp_path / "cache"
    out_dir = tmp_path / "out"
    cache_dir.mkdir()
    out_dir.mkdir()
    return cache_dir, out_dir


def test_query_cache_keys_include_business_type(tmp_path, monkeypatch, tmp_dirs):
    cache_dir, out_dir = tmp_dirs

    # Create two configs that differ only by business_type
    cfg_pharmacy = _write_cfg(tmp_path, business_type="pharmacy", cache_dir=cache_dir, output_dir=out_dir)
    cfg_clinic = _write_cfg(tmp_path, business_type="clinic", cache_dir=cache_dir, output_dir=out_dir)

    # Build orchestrators
    orch_pharmacy = PipelineOrchestrator(str(cfg_pharmacy))
    orch_clinic = PipelineOrchestrator(str(cfg_clinic))

    # Patch collector to produce deterministic results
    def fake_run_trial(query, location):
        return [{"name": "Foo Pharmacy", "address": "123 Main St"}]

    monkeypatch.setattr(orch_pharmacy, "collector", type("C", (), {"run_trial": staticmethod(fake_run_trial)})())
    monkeypatch.setattr(orch_clinic, "collector", type("C", (), {"run_trial": staticmethod(fake_run_trial)})())

    # Same query/location pair
    query = "independent pharmacy"
    location = "San Francisco, CA"

    # Execute once with each orchestrator to populate cache
    res1 = orch_pharmacy._execute_pharmacy_query(query, location)
    res2 = orch_clinic._execute_pharmacy_query(query, location)

    assert res1 and res2

    # If cache keys include business_type, two different cache files should exist
    cache_files = sorted(p.name for p in cache_dir.glob("*.json"))
    assert len(cache_files) == 2, f"expected 2 cache files, found {cache_files}"


def test_query_cache_hit_is_separate_by_business_type(tmp_path, monkeypatch, tmp_dirs):
    cache_dir, out_dir = tmp_dirs

    cfg_pharmacy = _write_cfg(tmp_path, business_type="pharmacy", cache_dir=cache_dir, output_dir=out_dir)
    cfg_clinic = _write_cfg(tmp_path, business_type="clinic", cache_dir=cache_dir, output_dir=out_dir)

    orch_pharmacy = PipelineOrchestrator(str(cfg_pharmacy))
    orch_clinic = PipelineOrchestrator(str(cfg_clinic))

    call_counter = {"calls": 0}

    def fake_run_trial(query, location):
        call_counter["calls"] += 1
        return [{"name": "Foo Pharmacy", "address": "123 Main St"}]

    monkeypatch.setattr(orch_pharmacy, "collector", type("C", (), {"run_trial": staticmethod(fake_run_trial)})())
    monkeypatch.setattr(orch_clinic, "collector", type("C", (), {"run_trial": staticmethod(fake_run_trial)})())

    query = "independent pharmacy"
    location = "San Francisco, CA"

    # First run to create both caches
    orch_pharmacy._execute_pharmacy_query(query, location)
    orch_clinic._execute_pharmacy_query(query, location)

    # Reset counter and run again, expecting both to hit their own cache without new API calls
    call_counter["calls"] = 0
    orch_pharmacy._execute_pharmacy_query(query, location)
    orch_clinic._execute_pharmacy_query(query, location)

    # No additional API calls should have been made if cache keys are separate
    assert call_counter["calls"] == 0
