"""
Caching layer for storing and retrieving API responses.
"""
import os
import time
import json
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, Optional, TypeVar, Generic, Union

T = TypeVar('T')

logger = logging.getLogger(__name__)

class CacheStats:
    """Cache statistics."""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.expired = 0
        self.evicted = 0
        self.size = 0

    def to_dict(self) -> Dict[str, int]:
        """Convert stats to dictionary."""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'expired': self.expired,
            'evicted': self.evicted,
            'size': self.size
        }


class CacheEntry(Generic[T]):
    """Cache entry with expiration support."""
    
    def __init__(self, value: T, ttl_seconds: float = 3600):
        """Initialize cache entry.
        
        Args:
            value: The value to cache
            ttl_seconds: Time to live in seconds
        """
        self.value = value
        self.expires_at = time.time() + ttl_seconds if ttl_seconds else None
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at


class Cache(Generic[T]):
    """In-memory cache with TTL and size-based eviction."""
    
    def __init__(self, name: str, cache_dir: str = '.cache', max_size: int = 1000, ttl: float = 3600):
        """Initialize cache.
        
        Args:
            name: Cache name (used for disk storage)
            cache_dir: Base directory for disk cache
            max_size: Maximum number of items in cache
            ttl: Default TTL in seconds
        """
        self.name = name
        self.cache_dir = os.path.join(cache_dir, name)
        self.max_size = int(max_size)  # Ensure max_size is an integer
        self.ttl = float(ttl)  # Ensure ttl is a float
        self._cache: Dict[str, CacheEntry[T]] = {}
        self.stats = CacheStats()
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_cache_file(self, key: str) -> str:
        """Get cache file path for key."""
        key_hash = hashlib.md5(key.encode('utf-8')).hexdigest()
        return os.path.join(self.cache_dir, f"{key_hash}.json")
    
    def _evict_if_needed(self):
        """Evict old entries if cache is full."""
        if len(self._cache) >= self.max_size:
            # Find oldest accessed entry
            oldest_key = min(
                self._cache.keys(),
                key=lambda k: self._cache[k].expires_at or float('inf')
            )
            self._cache.pop(oldest_key, None)
            self.stats.evicted += 1
            self.stats.size = len(self._cache)
    
    def set(self, key: str, value: T, ttl: Optional[float] = None) -> None:
        """Add an item to the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (defaults to self.ttl)
        """
        ttl_seconds = float(ttl) if ttl is not None else self.ttl
        self._evict_if_needed()
        self._cache[key] = CacheEntry(value, ttl_seconds)
        self.stats.size = len(self._cache)
        
        # Also save to disk
        cache_file = self._get_cache_file(key)
        item_to_persist = {
            "data": value,
            "expiry": self._cache[key].expires_at,
        }

        try:
            with open(cache_file, 'w') as f:
                json.dump(item_to_persist, f)
        except (IOError, json.JSONDecodeError) as e:
            logger.warning(f"Could not write to cache file {cache_file}: {e}")

    def get(self, key: str, default: Any = None) -> Optional[T]:
        """Get an item from the cache.
        
        Args:
            key: Cache key
            default: Default value if key not found
            
        Returns:
            Cached value or default if not found/expired
        """
        # Try memory cache first
        entry = self._cache.get(key)
        if entry:
            if not entry.is_expired():
                return entry.value
            self.stats.expired += 1
            del self._cache[key]
            self.stats.size = len(self._cache)
        
        # Try disk cache
        cache_file = Path(self._get_cache_file(key))
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                
                # Check if expired
                if data.get('expiry') and time.time() > data['expiry']:
                    os.remove(cache_file)
                    self.stats.expired += 1
                    self.stats.misses += 1
                    return default
                
                # Add back to memory cache
                remaining_ttl = data['expiry'] - time.time() if data.get('expiry') else None
                self._cache[key] = CacheEntry(data['data'], remaining_ttl)
                self.stats.hits += 1
                self.stats.size = len(self._cache)
                return data['data']
            except (json.JSONDecodeError, KeyError, OSError) as e:
                logger.warning(f"Corrupted cache file {cache_file}, removing it. Error: {e}")
                # Corrupted cache file, remove it
                try:
                    os.remove(cache_file)
                except OSError:
                    pass
        
        self.stats.misses += 1
        return default
    
    def invalidate(self, key: str) -> None:
        """Remove an item from the cache."""
        if key in self._cache:
            del self._cache[key]
            self.stats.size = len(self._cache)
        
        # Also remove from disk
        cache_file = self._get_cache_file(key)
        try:
            if os.path.exists(cache_file):
                os.remove(cache_file)
        except OSError:
            pass
    
    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self.stats = CacheStats()
        
        # Clear disk cache
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.json'):
                    os.remove(os.path.join(self.cache_dir, filename))
        except OSError:
            pass
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return self.stats.to_dict()
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in cache (not expired)."""
        return self.get(key, None) is not None


def get_cache(name: str, **kwargs) -> Cache:
    """Get or create a named cache.
    
    Args:
        name: Cache name
        **kwargs: Additional arguments for Cache constructor
        
    Returns:
        Cache instance
    """
    if not hasattr(get_cache, '_caches'):
        get_cache._caches = {}
    
    if name not in get_cache._caches:
        get_cache._caches[name] = Cache(name, **kwargs)
    
    return get_cache._caches[name]