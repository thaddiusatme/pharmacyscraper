from __future__ import annotations

# src/classification/cache.py
import functools
import json
import hashlib
from cachetools import TTLCache
from typing import Optional, List
import logging
from pathlib import Path

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

def save_to_cache(data: List[dict], cache_key: str, cache_dir: str) -> None:
    """Saves data to a JSON file in the cache directory."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    file_path = cache_path / f"{cache_key}.json"
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.debug(f"Saved data to cache: {file_path}")
    except (IOError, TypeError) as e:
        logger.warning(f"Failed to save cache file {file_path}: {e}")

def load_from_cache(cache_key: str, cache_dir: str) -> Optional[List[dict]]:
    """Loads data from a JSON file in the cache directory."""
    cache_path = Path(cache_dir)
    file_path = cache_path / f"{cache_key}.json"
    if not file_path.exists():
        return None
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        logger.debug(f"Loaded data from cache: {file_path}")
        return data
    except (IOError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to load cache file {file_path}: {e}")
        return None