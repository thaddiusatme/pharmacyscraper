"""Utility modules for the pharmacy scraper.

This package contains various utility modules that provide common functionality
used throughout the pharmacy scraper project.
"""

from .cache import Cache, cached

__all__ = [
    'Cache',
    'cached',
]