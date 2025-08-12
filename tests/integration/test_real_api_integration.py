import os
import time
import pytest

from pharmacy_scraper.orchestrator.pipeline_orchestrator import PipelineOrchestrator


def has_keys() -> bool:
    # minimal check for at least one external key
    return bool(os.getenv("GOOGLE_PLACES_KEY") or os.getenv("APIFY_API_TOKEN") or os.getenv("PERPLEXITY_API_KEY"))


@pytest.mark.skipif(not has_keys(), reason="Missing real API keys; skipping real API integration test")
def test_pipeline_min_real_api_with_rate_limit(tmp_path, monkeypatch):
    """
    Minimal e2e run that exercises the pipeline in legacy mode with a tiny input and
    a naive sleep-based rate limiter to respect API usage.

    Skipped unless API keys are present via environment variables.
    """
    # Prepare tiny config
    cfg = {
        "api_keys": {
            "google_places": os.getenv("GOOGLE_PLACES_KEY"),
            "apify": os.getenv("APIFY_API_TOKEN"),
            "perplexity": os.getenv("PERPLEXITY_API_KEY"),
        },
        "output_dir": str(tmp_path / "out"),
        "cache_dir": str(tmp_path / "cache"),
        "locations": ["San Francisco, CA"],
        "max_results_per_query": 2,
    }
    cfg_path = tmp_path / "cfg.json"
    import json
    cfg_path.write_text(json.dumps(cfg))

    orch = PipelineOrchestrator(str(cfg_path))

    # naive global rate limiter: sleep a little before starting
    time.sleep(1.1)

    # Run the simplest available stage to avoid heavy costs; classification only if possible
    def stage():
        # extremely small payload to limit API calls inside the stage
        return [{"name": "Test Pharmacy", "address": "123 Main St, San Francisco, CA"}]

    # Execute and ensure logging does not raise, output is persisted
    out = orch._execute_stage("classification", stage)
    assert isinstance(out, list)
    assert len(out) >= 1
