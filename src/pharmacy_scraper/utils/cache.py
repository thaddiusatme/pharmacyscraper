"""A unified caching solution for the pharmacy scraper.

This module provides a flexible caching system with both in-memory and file-based
storage options, TTL support, and decorators for easy function result caching.
"""

import json
import logging
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, TypeVar, Callable, Union
from functools import wraps
from cachetools import TTLCache

T = TypeVar('T')

logger = logging.getLogger(__name__)

class Cache:
    """A unified caching solution with file and in-memory support.
    
    This class provides a simple interface for caching data with optional
    persistence to disk. It supports TTL-based expiration and automatic
    cleanup of expired entries.
    """
    
    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        ttl: int = 3600,
        max_size_mb: Optional[float] = None,
        cleanup_interval: int = 300
    ) -> None:
        """Initialize the cache.
        
        Args:
            cache_dir: Directory for file-based cache. If None, only in-memory caching is used.
            ttl: Default time-to-live in seconds for cache entries.
            max_size_mb: Maximum size of the file cache in MB. Not yet implemented.
            cleanup_interval: How often to clean up expired entries, in seconds.
        """
        self.ttl = ttl
        self.cleanup_interval = cleanup_interval
        self.last_cleanup = time.time()
        
        # Initialize in-memory cache
        self.memory_cache = TTLCache(
            maxsize=1000,  # Default max items
            ttl=ttl
        )
        
        # Set up file cache if directory is provided
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None
    
    def _cleanup_if_needed(self) -> None:
        """Clean up expired cache entries if enough time has passed."""
        if time.time() - self.last_cleanup > self.cleanup_interval:
            self.cleanup()
            self.last_cleanup = time.time()
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache.
        
        Args:
            key: The cache key to look up.
            
        Returns:
            The cached value, or None if not found or expired.
        """
        # Try memory cache first
        try:
            if key in self.memory_cache:
                return self.memory_cache[key]
        except KeyError:
            pass
            
        # Try file cache if enabled
        if self.cache_dir:
            cache_file = self.cache_dir / f"{key}.json"
            if cache_file.exists():
                try:
                    with open(cache_file, 'r') as f:
                        data = json.load(f)
                    
                    # Check if entry has expired
                    if data.get('expires_at', float('inf')) > time.time():
                        # Update memory cache for faster access
                        self.memory_cache[key] = data['value']
                        return data['value']
                    
                    # Entry has expired, clean it up
                    try:
                        cache_file.unlink()
                    except OSError:
                        pass
                        
                except (json.JSONDecodeError, IOError) as e:
                    logger.warning(f"Error reading cache file {cache_file}: {e}")
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value in cache.
        
        Args:
            key: The cache key.
            value: The value to cache. Must be JSON-serializable for file cache.
            ttl: Optional TTL in seconds. Uses instance default if not specified.
        """
        ttl = ttl or self.ttl
        # Handle case where ttl is None - use a very large value (10 years)
        # or don't set expires_at at all
        expires_at = time.time() + (ttl if ttl is not None else 10 * 365 * 24 * 60 * 60)
        
        # Update memory cache
        self.memory_cache[key] = value
        
        # Update file cache if enabled
        if self.cache_dir:
            cache_file = self.cache_dir / f"{key}.json"
            try:
                with open(cache_file, 'w') as f:
                    json.dump({
                        'value': value,
                        'expires_at': expires_at,
                        'created_at': time.time(),
                        'ttl': ttl
                    }, f, indent=2)
            except (TypeError, IOError) as e:
                logger.warning(f"Error writing to cache file {cache_file}: {e}")
                raise
    
    def delete(self, key: str) -> None:
        """Delete a key from the cache."""
        # Remove from memory cache
        try:
            del self.memory_cache[key]
        except KeyError:
            pass
        
        # Remove from file cache
        if self.cache_dir:
            cache_file = self.cache_dir / f"{key}.json"
            try:
                if cache_file.exists():
                    cache_file.unlink()
            except OSError as e:
                logger.warning(f"Error deleting cache file {cache_file}: {e}")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        # Clear memory cache
        self.memory_cache.clear()
        
        # Clear file cache
        if self.cache_dir and self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    cache_file.unlink()
                except OSError as e:
                    logger.warning(f"Error deleting cache file {cache_file}: {e}")
    
    def cleanup(self) -> None:
        """Remove expired cache entries."""
        if not self.cache_dir or not self.cache_dir.exists():
            return
            
        current_time = time.time()
        removed = 0
        
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                
                if data.get('expires_at', 0) <= current_time:
                    cache_file.unlink()
                    removed += 1
                    
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Error during cache cleanup of {cache_file}: {e}")
                try:
                    cache_file.unlink()
                except OSError:
                    pass
        
        if removed > 0:
            logger.debug(f"Cleaned up {removed} expired cache entries")

def cached(cache: Cache, key_func: Optional[Callable[..., str]] = None):
    """Decorator to cache function results.
    
    Args:
        cache: A Cache instance to use for storage.
        key_func: Optional function to generate cache keys from function arguments.
                 If not provided, a default key will be generated.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func is not None:
                key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [func.__module__, func.__name__]
                if args:
                    key_parts.extend(str(arg) for arg in args)
                if kwargs:
                    key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                key = hashlib.md5("_".join(key_parts).encode()).hexdigest()
            
            # Try to get from cache
            cached_result = cache.get(key)
            if cached_result is not None:
                return cached_result
                
            # Call the function and cache the result
            result = func(*args, **kwargs)
            cache.set(key, result)
            return result
        return wrapper
    return decorator
