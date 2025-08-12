import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Iterable

# Keys that are standard on LogRecord; anything else from record.__dict__ is treated as extra
_STANDARD_RECORD_KEYS = {
    'name','msg','args','levelname','levelno','pathname','filename','module','exc_info','exc_text',
    'stack_info','lineno','funcName','created','msecs','relativeCreated','thread','threadName',
    'processName','process','message','asctime'
}


# Default fields to redact from payload if present
_DEFAULT_REDACT_FIELDS = {"api_key", "address", "email", "phone"}


def _utc_rfc3339() -> str:
    return datetime.now(timezone.utc).isoformat()


class _StructuredFilter(logging.Filter):
    """Filter that converts records into a JSON-serialized message with context.

    This filter looks up context bound to the logger emitting the record, so it works
    even when records propagate to ancestor loggers' handlers.
    """

    def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
        # Fetch the logger that created this record to read its bound contexts
        rec_logger = logging.getLogger(record.name)
        logger_context: Dict[str, Any] = getattr(rec_logger, "_logger_context", {})
        base_context: Dict[str, Any] = getattr(rec_logger, "_base_context", {})

        # Collect extra fields from the record (provided via `extra=` or adapters)
        extras: Dict[str, Any] = {
            k: v for k, v in record.__dict__.items() if k not in _STANDARD_RECORD_KEYS and not k.startswith("_")
        }

        payload: Dict[str, Any] = {}
        payload.update(base_context)
        payload.update(logger_context)
        payload.update(extras)

        payload.update(
            {
                "timestamp": _utc_rfc3339(),
                "level": record.levelname,
                "message": record.getMessage(),
            }
        )

        # Redact sensitive fields (top-level) if present
        for key in list(payload.keys()):
            if key in _DEFAULT_REDACT_FIELDS and key in payload:
                payload[key] = "REDACTED"

        # Replace the record message with JSON string
        record.msg = json.dumps(payload, ensure_ascii=False)
        record.args = ()
        return True


def _attach_filter_and_context(logger: logging.Logger, base_context: Optional[Dict[str, Any]] = None,
                               logger_context: Optional[Dict[str, Any]] = None) -> None:
    # Store contexts on the logger; copied into record by a wrapping filter attached to the logger
    setattr(logger, "_base_context", dict(base_context or {}))
    setattr(logger, "_logger_context", dict(logger_context or {}))

    # Ensure our structured filter is present on logger so even ad-hoc handlers output JSON
    have_filter = any(isinstance(f, _StructuredFilter) for f in logger.filters)
    if not have_filter:
        logger.addFilter(_StructuredFilter())


def bind_context(logger: logging.Logger, context: Dict[str, Any]) -> logging.Logger:
    """Return a child logger with additive structured context.

    This returns a distinct logger instance so callers can add different context without mutating the parent.
    """
    child_name = f"{logger.name}.ctx{abs(hash(frozenset(context.items())))%100000}"
    child = logging.getLogger(child_name)

    # Inherit level and ensure propagation so parent handlers receive records
    child.setLevel(logger.level)
    child.propagate = True

    # Merge contexts
    base = getattr(logger, "_base_context", {})
    parent_ctx = getattr(logger, "_logger_context", {})
    merged = {**parent_ctx, **context}
    _attach_filter_and_context(child, base_context=base, logger_context=merged)

    return child


def get_structured_logger(name: str, *, base_context: Optional[Dict[str, Any]] = None, level: int = logging.INFO) -> logging.Logger:
    """Create or configure a logger that emits JSON lines regardless of handler formatting.

    The JSON serialization happens in a logger-level filter, so even if a caller replaces handlers
    without setting a formatter, output remains structured.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # avoid double logging if root has handlers

    _attach_filter_and_context(logger, base_context=base_context or {}, logger_context={})

    # If no handler exists, attach a simple StreamHandler to stderr
    if not logger.handlers:
        handler = logging.StreamHandler()
        logger.addHandler(handler)

    return logger
