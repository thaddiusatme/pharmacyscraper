"""
Environment settings and validation utilities.

Provides a single place to verify required environment variables are present
before running any pipeline that makes external API calls.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Iterable, List, Tuple

import logging

logger = logging.getLogger(__name__)


def _resolve_apify_token() -> str | None:
    """Return APIFY token from either APIFY_TOKEN or APIFY_API_TOKEN."""
    return os.getenv("APIFY_TOKEN") or os.getenv("APIFY_API_TOKEN")


def get_required_env() -> Dict[str, str | None]:
    """Collect required environment variables and their current values.

    Returns a mapping of logical key name to its resolved value. For APIFY,
    either APIFY_TOKEN or APIFY_API_TOKEN is accepted and exposed under the
    APIFY_TOKEN logical key for simplicity.
    """
    return {
        "APIFY_TOKEN": _resolve_apify_token(),
        "GOOGLE_MAPS_API_KEY": os.getenv("GOOGLE_MAPS_API_KEY"),
        "PERPLEXITY_API_KEY": os.getenv("PERPLEXITY_API_KEY"),
    }


def validate_required_env(allow_missing: Iterable[str] | None = None) -> bool:
    """Validate that required environment variables are present.

    Args:
        allow_missing: Optional iterable of logical keys to allow missing.
                       Useful for dry runs or partial feature sets.

    Returns:
        True if validation passes; False otherwise.
    """
    allow_set = set(allow_missing or [])
    required = get_required_env()

    missing = [k for k, v in required.items() if not v and k not in allow_set]
    if missing:
        logger.error(
            "Missing required environment variables: %s",
            ", ".join(sorted(missing)),
        )
        logger.error("Please set them in your environment or your .env file.")
        return False

    # Additional guidance for APIFY dual variable situation
    if not os.getenv("APIFY_TOKEN") and os.getenv("APIFY_API_TOKEN"):
        logger.info(
            "Using APIFY_API_TOKEN for APIFY_TOKEN. Consider setting APIFY_TOKEN for consistency."
        )

    return True


def exit_if_invalid_env(allow_missing: Iterable[str] | None = None) -> None:
    """Exit the process with a non-zero status if required env vars are missing."""
    if not validate_required_env(allow_missing=allow_missing):
        # Use explicit message for clarity in CLI contexts
        print(
            "Error: Required environment variables are missing. See logs above for details.",
            file=sys.stderr,
        )
        sys.exit(1)
