import json
import logging
from contextlib import contextmanager

import pytest

from pharmacy_scraper.observability.logging import (
    get_structured_logger,
    bind_context,
)


@contextmanager
def capture_stream_logger(logger: logging.Logger):
    import io
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    # Ensure we use the JSON formatter from our logger
    for h in list(logger.handlers):
        logger.removeHandler(h)
    logger.addHandler(handler)
    try:
        yield stream, handler
    finally:
        logger.removeHandler(handler)


def test_json_log_contains_fields():
    logger = get_structured_logger("test.logger", base_context={"run_id": "r123", "stage": "classification"})
    with capture_stream_logger(logger) as (stream, handler):
        # Bind additional context and log
        log = bind_context(logger, {"component": "unit", "event": "stage_start"})
        log.info("starting stage", extra={"result_count": 0})
        handler.flush()
        data = stream.getvalue().strip().splitlines()
        assert len(data) == 1
        payload = json.loads(data[0])
        # Required fields
        assert payload["run_id"] == "r123"
        assert payload["stage"] == "classification"
        assert payload["component"] == "unit"
        assert payload["event"] == "stage_start"
        assert payload["message"] == "starting stage"
        assert payload["level"] == "INFO"
        assert "timestamp" in payload
        # Extra propagated
        assert payload["result_count"] == 0


def test_bind_context_is_additive():
    logger = get_structured_logger("test.logger", base_context={"run_id": "r123"})
    l2 = bind_context(logger, {"stage": "dedup"})
    l3 = bind_context(l2, {"component": "adapter"})
    with capture_stream_logger(l3) as (stream, handler):
        l3.warning("hello")
        handler.flush()
        payload = json.loads(stream.getvalue().strip())
        assert payload["run_id"] == "r123"
        assert payload["stage"] == "dedup"
        assert payload["component"] == "adapter"
        assert payload["level"] == "WARNING"
