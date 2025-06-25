from __future__ import annotations

# src/classification/cache.py
import functools
import json
import hashlib
from cachetools import TTLCache
from typing import Optional
import logging

logger = logging.getLogger(__name__)

_cache = TTLCache(maxsize=1024, ttl=3600)

class Cache:  # simple wrapper used in unit tests
    """A lightweight in-memory cache with the expected get/set API.

    The real implementation previously lived in a different module but some
    legacy tests still import ``src.classification.cache.Cache``.  This shim
    keeps those tests working without changing their code.
    """

    def __init__(self, ttl: Optional[int] = None, maxsize: int = 1024):
        ttl = ttl or 3600
        self._store = TTLCache(maxsize=maxsize, ttl=ttl)

    def get(self, key, default=None):  # noqa: D401 simple proxy
        return self._store.get(key, default)

    def set(self, key, value, ttl: Optional[int] = None):
        self._store[key] = value

    def clear(self):
        self._store.clear()

def _generate_key(*args, **kwargs):
    """Generate a cache key from the function's arguments."""
    try:
        # Use JSON to serialize arguments for a consistent key
        key_data = json.dumps((args, kwargs), sort_keys=True).encode('utf-8')
        return hashlib.sha256(key_data).hexdigest()
    except TypeError:
        # Fallback for non-serializable types
        return repr((args, kwargs))

def cache_wrapper(func):
    """A decorator to cache function results."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = _generate_key(args, kwargs)
        if key in _cache:
            logger.info(f"Cache hit for {func.__name__} with key {key}")
            return _cache[key]
        
        logger.info(f"Cache miss for {func.__name__} with key {key}, calling function.")
        result = func(*args, **kwargs)
        _cache[key] = result
        return result
    
    # Add a helper to clear the cache, useful for testing
    def clear_cache():
        _cache.clear()
    
    wrapper.clear_cache = clear_cache
    return wrapper