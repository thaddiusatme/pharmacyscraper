import json
import logging
from contextlib import contextmanager

import pytest

from pharmacy_scraper.observability.logging import get_structured_logger, bind_context
from pharmacy_scraper.classification.classifier import Classifier, clear_classification_cache
from pharmacy_scraper.orchestrator.pipeline_orchestrator import PipelineOrchestrator


@contextmanager
def capture_logger(logger: logging.Logger):
    import io
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    # Remove other handlers for isolation
    for h in list(logger.handlers):
        logger.removeHandler(h)
    logger.addHandler(handler)
    try:
        yield stream, handler
    finally:
        logger.removeHandler(handler)


def _read_lines(stream):
    return [json.loads(line) for line in stream.getvalue().strip().splitlines() if line.strip()]


def test_redaction_defaults():
    logger = get_structured_logger("obs.test", base_context={"run_id": "r1"})
    with capture_logger(logger) as (stream, handler):
        logger.info("sensitive", extra={"address": "123 Main", "email": "a@b.com", "phone": "555", "api_key": "secret"})
        handler.flush()
        rec = _read_lines(stream)[0]
        assert rec["address"] == "REDACTED"
        assert rec["email"] == "REDACTED"
        assert rec["phone"] == "REDACTED"
        assert rec["api_key"] == "REDACTED"


def test_classifier_cache_logging_sequence():
    clear_classification_cache()
    clf = Classifier(client=None)
    base_logger = get_structured_logger("pharmacy_scraper.classification.classifier", base_context={"run_id": "r-cache"})
    with capture_logger(base_logger) as (stream, handler):
        # First call: miss + store
        clf.classify_pharmacy({"name": "Acme Pharmacy"}, use_llm=False)
        # Second call: hit
        clf.classify_pharmacy({"name": "Acme Pharmacy"}, use_llm=False)
        handler.flush()
        lines = _read_lines(stream)
        events = [l.get("event") for l in lines]
        # Expect at least one miss, one store, and one hit
        assert "cache_miss" in events
        assert "cache_store" in events
        assert "cache_hit" in events


def test_execute_stage_logging_without_clients(tmp_path, monkeypatch):
    # Create minimal config with plugin_mode true to avoid initializing external clients
    cfg = {
        "plugin_mode": True,
        "api_keys": {},
        "output_dir": str(tmp_path / "out"),
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))

    orch = PipelineOrchestrator(str(cfg_path))
    # Install a run logger with fixed run_id
    orch._run_logger = get_structured_logger("pharmacy_scraper.orchestrator.pipeline_orchestrator", base_context={"run_id": "r-orch"})

    # Use a simple stage function
    def stage():
        return [1, 2, 3]

    base_logger = orch._run_logger
    with capture_logger(base_logger) as (stream, handler):
        result = orch._execute_stage("classification", stage)
        assert result == [1, 2, 3]
        handler.flush()
        lines = _read_lines(stream)
        events = [l.get("event") for l in lines]
        assert "stage_start" in events
        assert "stage_completed" in events
        # Ensure stage field is present on stage logs
        stage_lines = [l for l in lines if l.get("event") in {"stage_start", "stage_completed"}]
        assert all(l.get("stage") == "classification" for l in stage_lines)
