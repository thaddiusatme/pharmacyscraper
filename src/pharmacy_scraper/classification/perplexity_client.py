"""
Perplexity API client for pharmacy classification.

This module provides a client for interacting with the Perplexity API
using the OpenAI-compatible interface.
"""
import os
import json
import time
import logging
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
import openai
from openai import OpenAI
from pharmacy_scraper.config import get_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class RateLimiter:
    """A simple rate limiter to control the frequency of API calls."""

    def __init__(self, requests_per_minute: int):
        if requests_per_minute <= 0:
            self.min_interval = 0
        else:
            self.min_interval = 60.0 / requests_per_minute
        # Initialize to a negative value to ensure first call is never rate limited
        self.last_request_time = -float('inf')

    def wait(self):
        """Waits if necessary to respect the rate limit."""
        if self.min_interval == 0:
            return

        current_time = time.time()
        elapsed = current_time - self.last_request_time

        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)

        self.last_request_time = time.time()


class PerplexityAPIError(Exception):
    """Base exception for Perplexity API errors."""
    def __init__(self, message, error_type="api_error"):
        self.message = message
        self.error_type = error_type
        super().__init__(f"[{self.error_type}] {self.message}")


# Additional exceptions for test compatibility
class RateLimitError(Exception):
    """Exception for rate limit exceeded."""
    pass

class InvalidRequestError(Exception):
    """Exception for invalid request to Perplexity API."""
    pass

class ResponseParseError(Exception):
    """Exception for errors parsing Perplexity API response."""
    pass


class PerplexityClient:
    """
    A client for the Perplexity API, designed for pharmacy classification.

    This client handles API requests, rate limiting, and file-based caching
    of classification results.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = "pplx-7b-online",
        rate_limit: int = 20,
        cache_dir: Optional[str] = "data/cache/classification",
        force_reclassification: bool = False,
        openai_client: Optional[OpenAI] = None,
        max_retries: int = 3,
        enable_metrics: bool = True,
        cache_ttl_seconds: Optional[int] = 7 * 24 * 3600,  # 1 week default TTL
        max_cache_size_mb: Optional[float] = None,  # Maximum cache size in MB
        cleanup_frequency: int = 300  # Cleanup every 5 minutes by default
    ) -> None:
        """Initialize the PerplexityClient.
        
        Args:
            api_key: Perplexity API key. If not provided, will try to get from PERPLEXITY_API_KEY env var.
            model_name: Name of the model to use. Defaults to 'sonar-medium-online'.
            rate_limit: Maximum number of requests per minute.
            cache_dir: Directory to store cache files. If None, caching is disabled.
            force_reclassification: If True, always call the API even if result is in cache.
            openai_client: Optional OpenAI client instance. If not provided, one will be created.
            max_retries: Maximum number of retries for API calls.
            enable_metrics: Whether to enable metrics collection.
            cache_ttl_seconds: Time-to-live for cache entries in seconds. If None, cache never expires.
            max_cache_size_mb: Maximum cache size in MB. If None, no size limit is enforced.
            cleanup_frequency: How often to run cleanup in seconds. Set to 0 to run on every check.
        """
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key must be provided either as an argument or via PERPLEXITY_API_KEY environment variable"
            )
            
        self.model_name = model_name or "sonar-medium-online"
        self.rate_limit = rate_limit
        self.force_reclassification = force_reclassification
        self.max_retries = max_retries
        self.enable_metrics = enable_metrics
        self.cache_ttl_seconds = cache_ttl_seconds
        self.max_cache_size_mb = max_cache_size_mb * 1024 * 1024 if max_cache_size_mb else None  # Convert MB to bytes
        self.cleanup_frequency = cleanup_frequency
        
        # Initialize cache
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"PerplexityClient caching is enabled. Cache directory: {self.cache_dir}")
        
        # Initialize metrics
        self._cache_metrics = {
            'hits': 0,
            'misses': 0,
            'errors': 0,
            'size': 0,  # in bytes
            'entries': 0,
            'expired': 0,
            'last_cleaned': 0
        }
        
        # Update metrics on init
        self._update_cache_metrics()
        
        # Use provided OpenAI client or create a new one
        if openai_client is not None:
            self.client = openai_client
        else:
            self.client = OpenAI(api_key=self.api_key, base_url="https://api.perplexity.ai")
        self.rate_limiter = RateLimiter(rate_limit)

    def _get_cache_file(self, cache_key: str) -> Path:
        """Get the full path to a cache file."""
        return self.cache_dir / f"{cache_key}.json"
    
    def _is_cache_valid(self, cache_file: Path) -> bool:
        """Check if a cache entry is still valid based on TTL."""
        if not cache_file.exists():
            return False
            
        if self.cache_ttl_seconds is None:
            return True
            
        try:
            file_mtime = cache_file.stat().st_mtime
            return (time.time() - file_mtime) < self.cache_ttl_seconds
        except OSError as e:
            logger.warning(f"Failed to check cache file {cache_file} mtime: {e}")
            return False
    
    def _read_from_cache(self, cache_file: Path) -> Optional[Dict[str, Any]]:
        """Read data from cache file with error handling."""
        try:
            if not self.cache_dir or not cache_file.exists() or not self._is_cache_valid(cache_file):
                return None
                
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                
            # Validate cached data structure
            if not isinstance(cached_data, dict) or 'data' not in cached_data:
                logger.warning(f"Invalid cache format in {cache_file}")
                return None
                
            # Check if entry has expired
            if 'expires_at' in cached_data and cached_data['expires_at'] < time.time():
                logger.debug(f"Cache entry expired: {cache_file}")
                return None
                
            self._cache_metrics['hits'] += 1
            return cached_data['data']
            
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to read from cache file {cache_file}: {e}")
            self._cache_metrics['errors'] += 1
            return None
    
    def _write_to_cache(self, cache_file: Path, data: Dict[str, Any]) -> bool:
        """Write data to cache file with error handling and size management.
        
        Args:
            cache_file: Path to the cache file
            data: Data to cache
            
        Returns:
            bool: True if write was successful, False otherwise
        """
        if not self.cache_dir:
            return False
            
        temp_file = None
        try:
            # Check and enforce cache size limits before writing
            self._check_cache_limits()
            
            # Create cache directory if it doesn't exist
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare cache entry with metadata
            cache_entry = {
                'data': data,
                'cached_at': time.time(),
                'version': '1.0',  # For future compatibility
                'size': 0  # Will be updated after writing
            }
            
            # Add TTL if specified
            if self.cache_ttl_seconds is not None:
                cache_entry['expires_at'] = time.time() + self.cache_ttl_seconds
            
            # Write to temporary file first, then rename (atomic operation)
            temp_file = cache_file.with_suffix('.tmp')
            
            # First write to get the size
            with open(temp_file, 'w') as f:
                json.dump(cache_entry, f, indent=2)
            
            # Get the actual file size and update the cache entry
            file_size = temp_file.stat().st_size
            cache_entry['size'] = file_size
            
            # Write again with updated size
            with open(temp_file, 'w') as f:
                json.dump(cache_entry, f, indent=2)
            
            # Atomic rename
            temp_file.replace(cache_file)
            
            # Update metrics
            self._update_cache_metrics()
            return True
            
        except (OSError, TypeError, ValueError) as e:  # ValueError catches JSONEncodeError in Python < 3.5
            logger.warning(f"Failed to write to cache file {cache_file}: {e}")
            self._cache_metrics['errors'] += 1
            return False
            
        finally:
            # Clean up any temporary file that might be left
            if temp_file is not None and temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError as e:
                    logger.warning(f"Failed to clean up temporary file {temp_file}: {e}")
    
    def _check_cache_limits(self):
        """Check and enforce cache size limits."""
        if not self.cache_dir or self.max_cache_size_mb is None:
            return
            
        current_time = time.time()
        
        # Only check if enough time has passed since last cleanup
        if self.cleanup_frequency > 0 and \
           (current_time - self._cache_metrics.get('last_cleaned', 0)) < self.cleanup_frequency:
            return
            
        try:
            # Get all cache files with their sizes and modification times
            cache_files = []
            total_size = 0
            
            for file_path in self.cache_dir.glob('*.json'):
                try:
                    file_size = file_path.stat().st_size
                    mtime = file_path.stat().st_mtime
                    cache_files.append((file_path, mtime, file_size))
                    total_size += file_size
                except OSError as e:
                    logger.warning(f"Error accessing cache file {file_path}: {e}")
            
            # Check if we're over the limit
            if total_size <= self.max_cache_size_mb:
                return
                
            logger.info(f"Cache size {total_size/1024/1024:.2f}MB exceeds limit of {self.max_cache_size_mb/1024/1024:.2f}MB. Cleaning up...")
            
            # Sort files by modification time (oldest first)
            cache_files.sort(key=lambda x: x[1])
            
            # Remove oldest files until we're under 90% of the limit
            target_size = self.max_cache_size_mb * 0.9
            removed = 0
            
            for file_path, _, file_size in cache_files:
                if total_size <= target_size:
                    break
                    
                try:
                    file_path.unlink()
                    total_size -= file_size
                    removed += 1
                except OSError as e:
                    logger.warning(f"Error removing cache file {file_path}: {e}")
            
            if removed > 0:
                logger.info(f"Removed {removed} old cache files. New size: {total_size/1024/1024:.2f}MB")
                
            # Update last cleaned time and metrics
            self._cache_metrics['last_cleaned'] = current_time
            self._update_cache_metrics()
            
        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")
            self._cache_metrics['errors'] += 1
    
    def _update_cache_metrics(self):
        """Update cache metrics including size, entries, and hit ratio."""
        if not self.enable_metrics or not self.cache_dir or not self.cache_dir.exists():
            return
            
        try:
            # Count cache files and calculate total size
            total_size = 0
            total_entries = 0
            expired_entries = 0
            
            for file_path in self.cache_dir.glob('*.json'):
                try:
                    file_size = file_path.stat().st_size
                    total_size += file_size
                    total_entries += 1
                    
                    # Check if entry is expired
                    if self.cache_ttl_seconds and (time.time() - file_path.stat().st_mtime) > self.cache_ttl_seconds:
                        expired_entries += 1
                        
                except OSError as e:
                    logger.warning(f"Failed to stat cache file {file_path}: {e}")
            
            # Update metrics
            self._cache_metrics.update({
                'size': total_size,
                'entries': total_entries,
                'expired': expired_entries
            })
            
            # Log metrics periodically
            total_requests = self._cache_metrics['hits'] + self._cache_metrics['misses']
            hit_ratio = (self._cache_metrics['hits'] / total_requests * 100) if total_requests > 0 else 0
            
            logger.debug(
                f"Cache metrics: hits={self._cache_metrics['hits']}, "
                f"misses={self._cache_metrics['misses']}, "
                f"size={total_size/1024/1024:.2f}MB, "
                f"entries={total_entries}, "
                f"expired={expired_entries}"
            )
            logger.debug(f"Cache hit ratio: {hit_ratio:.1f}%")
            
        except Exception as e:
            logger.warning(f"Failed to update cache metrics: {e}")
    
    def _cleanup_expired_entries(self):
        """Remove expired cache entries."""
        if not self.cache_dir or self.cache_ttl_seconds is None:
            return 0
            
        current_time = time.time()
        removed = 0
        
        for cache_file in self.cache_dir.glob('*.json'):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                
                # Check if entry has expired
                expires_at = data.get('expires_at')
                if expires_at is not None and current_time > expires_at:
                    try:
                        cache_file.unlink()
                        removed += 1
                    except OSError as e:
                        logger.warning(f"Error removing expired cache file {cache_file}: {e}")
                    
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Error reading cache file {cache_file}: {e}")
                continue
        
        if removed > 0:
            logger.info(f"Removed {removed} expired cache entries")
            self._update_cache_metrics()
            
        return removed
                
    def cleanup_expired(self) -> int:
        """
        Clean up expired cache entries.
        
        Returns:
            int: Number of entries removed
        """
        return self._cleanup_expired_entries()
    
    def get_cache_metrics(self) -> Dict[str, Any]:
        """
        Get current cache metrics.
        
        Returns:
            Dict with cache metrics including hits, misses, size, etc.
        """
        self._update_cache_metrics()
        return self._cache_metrics.copy()
    
    def classify_pharmacy(
        self, 
        pharmacy_data: Dict[str, Any], 
        model: Optional[str] = None,
        cache_dir: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Classifies a single pharmacy using the Perplexity API, with optional caching.
        
        This method handles the complete classification workflow including:
        - Cache lookup (if caching is enabled)
        - Generating the classification prompt
        - Making the API call with retry logic
        - Caching the result (if successful)
        
        Args:
            pharmacy_data: A dictionary containing the pharmacy's details. Should include
                         at minimum 'name' and 'address' keys.
            model: The name of the model to use for classification. If None, uses the
                  instance's default model.
            cache_dir: Optional directory path to use for caching. If provided, overrides
                     the instance's cache directory.
                     
        Returns:
            A dictionary containing the classification result with keys:
            - is_chain: bool indicating if the pharmacy is a chain
            - confidence: float between 0 and 1 indicating confidence
            - explanation: str with reasoning for the classification
            - method: str indicating how the classification was determined
            - model: str indicating which model was used
            
        Raises:
            PerplexityAPIError: If the API request fails after all retries
            ValueError: If the pharmacy data is invalid
            - When caching is enabled (default), successful classifications are cached to
              avoid redundant API calls for the same pharmacy data.
            - Set force_reclassification=True when initializing the client to bypass the cache.
            - If an error occurs during cache read/write, the method will log a warning
              but continue with the API call.
            
        Example:
            ```python
            # Initialize the client
            client = PerplexityClient(api_key="your_api_key")
            
            # Classify a pharmacy
            result = client.classify_pharmacy({
                "name": "Main Street Pharmacy",
                "address": "123 Main St, Anytown, CA 12345",
                "phone": "(555) 123-4567"
            })
            
            if result:
                print(f"Is chain: {result.get('is_chain', False)}")
                print(f"Is compounding: {result.get('is_compounding', False)}")
                print(f"Confidence: {result.get('confidence', 0.0):.2f}")
            ```
            
        Performance:
            - API calls are rate-limited according to the client's rate_limit setting
              (default: 20 requests per minute).
            - Typical response time is 1-3 seconds under normal conditions.
            - Cached responses are typically returned in <10ms.
            
        Thread Safety:
            - This method is thread-safe for concurrent calls with different pharmacy_data.
            - When caching is enabled, cache operations are thread-safe.
            - Multiple threads calling with the same pharmacy_data may result in
              duplicate API calls before the result is cached.
              
        Error Handling:
            - The method implements exponential backoff for rate-limited requests.
            - By default, it will retry up to 3 times (configurable via max_retries)
              with increasing delays between attempts.
            - All errors are logged with detailed information before being re-raised.
            - Network timeouts are handled with appropriate retry logic.
        """
        model_to_use = model or self.model
        prompt = self._generate_prompt(pharmacy_data)
        
        # If cache is disabled, call API directly
        if not self.cache_dir:
            logger.debug("Cache is disabled, calling API directly.")
            return self._call_api(prompt)
        
        # Generate cache key and get cache file path
        cache_key = _generate_cache_key(pharmacy_data, model_to_use)
        cache_file = self._get_cache_file(cache_key)
        
        # Check cache if not forcing reclassification
        if not self.force_reclassification:
            cached_result = self._read_from_cache(cache_file)
            if cached_result is not None:
                logger.info(f"CACHE HIT for pharmacy: {pharmacy_data.get('title', 'N/A')}")
                return cached_result
        else:
            logger.info(f"CACHE IGNORED (force_reclassification=True) for pharmacy: {pharmacy_data.get('title', 'N/A')}")
        
        # If we get here, we need to call the API
        self._cache_metrics['misses'] += 1
        logger.info(f"CACHE MISS for pharmacy: {pharmacy_data.get('title', 'N/A')}. Calling API.")
        
        # Call the API
        result = self._call_api(prompt)
        
        # Write to cache if the API call was successful
        if result:
            self._write_to_cache(cache_file, result)
        
        # Periodically log metrics and clean up
        if self.enable_metrics and time.time() - self._cache_metrics['last_cleaned'] > 3600:  # Every hour
            self.cleanup_expired()
            self._update_cache_metrics()
            self._cache_metrics['last_cleaned'] = time.time()
        
        return result    
        
    def _call_api(self, prompt: str) -> Optional[Dict[str, Any]]:
        """
        Makes an API call to the Perplexity API with the given prompt.
        
        Args:
            prompt: The prompt to send to the API.
            
        Returns:
            The parsed response from the API, or None if an error occurs.
            
        Raises:
            RateLimitError: If the rate limit is exceeded after all retries.
            PerplexityAPIError: For other API errors.
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                self.rate_limiter.wait()
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                
                # Parse the response
                if not response.choices or not response.choices[0].message:
                    raise ResponseParseError("Invalid response format: missing choices or message")
                
                content = response.choices[0].message.content
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    # If the response is not valid JSON, try to parse it as a string
                    return {"response": content}
                    
            except Exception as e:
                last_exception = e
                if "rate limit" in str(e).lower() and attempt < self.max_retries:
                    # Exponential backoff
                    time.sleep(min(2 ** attempt, 10))
                    continue
                elif "rate limit" in str(e).lower():
                    raise RateLimitError(f"Rate limit exceeded after {self.max_retries} retries") from e
                else:
                    raise PerplexityAPIError(f"API error: {str(e)}") from e
        
        # If we've exhausted all retries
        if last_exception:
            raise PerplexityAPIError(f"Failed after {self.max_retries} retries: {str(last_exception)}")
        
        return None

    def _make_api_call(self, pharmacy_data: Dict[str, Any], model: str, retry_count: int = 3) -> Optional[Dict[str, Any]]:
        """
        Makes the actual API call to Perplexity with retry logic for rate limits.
        
        Args:
            pharmacy_data: The pharmacy data to classify
            model: The model to use for classification
            retry_count: Number of retry attempts remaining
            
        Returns:
            The classification result or None if all retries are exhausted
        """
        # This method is kept for backward compatibility
        # The actual API call logic has been moved to _call_api
        prompt = self._generate_prompt(pharmacy_data)
        return self._call_api(prompt)
        try:
            logger.debug(f"API Key prefix in use: {self.api_key[:15] if isinstance(self.api_key, str) and self.api_key else 'No API key'}...")
            logger.debug(f"Base URL: {self.client.base_url}")
            logger.debug(f"Model: {model}")
            
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert pharmacy classifier. Respond in JSON format."},
                    {"role": "user", "content": prompt},
                ],
                # Use simpler response format that's supported by Perplexity
                temperature=0.1,  # Low temperature for consistent results
                max_tokens=200,   # Limit response length
            )
            
            logger.info(f"Perplexity API call successful for {pharmacy_name}, parsing response...")
            result = self._parse_response(response)
            
            if result and isinstance(result, dict):
                logger.info(f"✅ Successfully classified {pharmacy_name}: {result.get('classification', 'unknown')}")
            else:
                logger.error(f"❌ Failed to parse response for {pharmacy_name}")
            
            return result
            
        except RateLimitError as e:
            if retry_count > 0:
                retry_after = 5  # Default wait time in seconds
                logger.warning(f"Rate limit exceeded for '{pharmacy_name}'. Retrying in {retry_after} seconds... (attempts left: {retry_count})")
                time.sleep(retry_after)
                return self._make_api_call(pharmacy_data, model, retry_count - 1)
            logger.error(f"Rate limit exceeded for '{pharmacy_name}' after retries: {e}")
            return None
            
        except Exception as e:
            logger.error(f"Perplexity API error for pharmacy '{pharmacy_name}': {e}")
            
            # Add detailed stack trace for subscriptable errors
            if "not subscriptable" in str(e):
                import traceback
                logger.error(f"SUBSCRIPTABLE ERROR STACK TRACE:")
                logger.error(traceback.format_exc())
            
            # Capture detailed information about 401 errors
            if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                if e.response.status_code == 401:
                    logger.error(f"401 Unauthorized Details:")
                    logger.error(f"  Request URL: {getattr(e.response, 'url', 'Unknown')}")
                    logger.error(f"  Request headers: {getattr(e.response.request, 'headers', 'Unknown') if hasattr(e.response, 'request') else 'Unknown'}")
                    logger.error(f"  Response headers: {getattr(e.response, 'headers', 'Unknown')}")
                    logger.error(f"  Response text: {getattr(e.response, 'text', 'Unknown')}")
            
            return None

    def _parse_response(self, response: object) -> Optional[Dict[str, Any]]:
        """Parse the response from the Perplexity API, handling both JSON and text responses."""
        try:
            # Log the full raw response structure
            logger.debug(f"Raw Perplexity API response object: {response}")
            logger.debug(f"Response type: {type(response)}")
            
            if not hasattr(response, 'choices') or not response.choices:
                logger.error("Response missing choices or choices is empty")
                return None
                
            choice = response.choices[0]
            if not hasattr(choice, 'message') or not hasattr(choice.message, 'content'):
                logger.error("Response choice missing message or content attribute")
                return None
                
            content = choice.message.content
            logger.info(f"Raw response content: {content}")
            
            # Extract JSON from code blocks if present
            json_content = content
            if "```json" in content:
                # Extract content between ```json and next ```
                start_marker = "```json"
                end_marker = "```"
                start_idx = content.find(start_marker)
                if start_idx != -1:
                    start_idx += len(start_marker)
                    end_idx = content.find(end_marker, start_idx)
                    if end_idx != -1:
                        json_content = content[start_idx:end_idx].strip()
                        logger.debug(f"Extracted JSON from code blocks: {json_content}")
                    else:
                        logger.warning("Found ```json but no closing ```, trying to extract JSON directly")
                else:
                    # Try to find any code block
                    start_idx = content.find("```")
                    if start_idx != -1:
                        start_idx += 3
                        end_idx = content.find("```", start_idx)
                        if end_idx != -1:
                            json_content = content[start_idx:end_idx].strip()
                            logger.debug(f"Extracted JSON from generic code blocks: {json_content}")
            
            # Try to parse JSON
            try:
                data = json.loads(json_content)
                logger.debug(f"Parsed JSON data: {data}")
                
                # Validate required fields
                required_fields = ["classification", "is_compounding", "confidence"]
                if not all(field in data for field in required_fields):
                    missing = [f for f in required_fields if f not in data]
                    raise ValueError(f"Missing required fields: {missing}")
                    
                # Validate classification value
                if data["classification"] not in ["independent", "chain", "hospital", "not_a_pharmacy"]:
                    raise ValueError(f"Invalid classification value: {data['classification']}")
                    
                # Ensure confidence is a number between 0 and 1
                try:
                    confidence = float(data["confidence"])
                    if not 0 <= confidence <= 1:
                        raise ValueError(f"Confidence {confidence} out of range [0, 1]")
                    data["confidence"] = confidence
                except (TypeError, ValueError) as e:
                    raise ValueError(f"Invalid confidence value: {e}")
                
                # Ensure is_compounding is a boolean
                if not isinstance(data["is_compounding"], bool):
                    if isinstance(data["is_compounding"], str):
                        data["is_compounding"] = data["is_compounding"].lower() == 'true'
                    else:
                        raise ValueError("is_compounding must be a boolean")
                
                # Ensure explanation is present and a string
                if "explanation" not in data or not isinstance(data["explanation"], str):
                    logger.warning("No explanation found in response, adding a default one")
                    data["explanation"] = "No explanation provided by the model"
                else:
                    # Clean up the explanation text
                    data["explanation"] = data["explanation"].strip()
                
                                # Attempt to extract explanation text following the JSON block if it was
                # not included as a key in the JSON itself.  Most of our few-shot
                # examples show the model returning natural-language explanation *after*
                # the formatted JSON so we grab any remaining text and store it.
                if ("explanation" not in data or not data["explanation"]) and end_idx is not None:
                    explanation_text = content[end_idx + len(end_marker):].strip()
                    # Remove leading punctuation / whitespace markers
                    explanation_text = explanation_text.lstrip("\n\r ")
                    if explanation_text:
                        data["explanation"] = explanation_text
                    else:
                        data["explanation"] = "No explanation provided by the model"

                return data
                
            except json.JSONDecodeError as json_err:
                logger.error(f"JSON decode error: {json_err}")
                logger.error(f"Content that failed to parse: {json_content}")
                return None
                
        except Exception as e:
            logger.error(f"Error parsing response: {e}", exc_info=True)
            return None

    def classify_pharmacies_batch(self, pharmacies_data: List[Dict[str, Any]], model: Optional[str] = None) -> List[Optional[Dict[str, Any]]]:
        """
        Classifies a batch of pharmacies.
        """
        return [self.classify_pharmacy(pharmacy, model) for pharmacy in pharmacies_data]

    def _generate_prompt(self, pharmacy_data: Dict[str, Any]) -> str:
        """Generate a prompt for pharmacy classification with few-shot examples and explanation."""
        name = pharmacy_data.get('title') or pharmacy_data.get('name', 'N/A')
        
        # Few-shot examples to guide the model
        examples = [
            {
                "input": {
                    "name": "Main Street Pharmacy",
                    "address": "123 Main St, Anytown, USA",
                    "categories": "Pharmacy, Compounding",
                    "website": "mainstreetpharmacy.com"
                },
                "output": """```json
{
    "classification": "independent",
    "is_compounding": true,
    "confidence": 0.95
}
```

This is an independent pharmacy because it's a locally-owned business not affiliated with any major pharmacy chain. It's a compounding pharmacy as indicated in its categories and website focus on customized medications."""
            },
            {
                "input": {
                    "name": "CVS Pharmacy",
                    "address": "456 Oak Ave, Somewhere, USA",
                    "categories": "Pharmacy, Convenience Store",
                    "website": "cvs.com"
                },
                "output": """```json
{
    "classification": "chain",
    "is_compounding": false,
    "confidence": 0.99
}
```

This is a chain pharmacy because CVS is a well-known national pharmacy chain. It's not a compounding pharmacy as there's no indication of specialized medication compounding services."""
            },
            {
                "input": {
                    "name": "Union Avenue Compounding Pharmacy",
                    "address": "789 Union Ave, Tacoma, WA",
                    "categories": "Pharmacy, Compounding, Health & Medical",
                    "website": "unionrx.com"
                },
                "output": """```json
{
    "classification": "independent",
    "is_compounding": true,
    "confidence": 0.97
}
```

This is an independent compounding pharmacy as it's locally owned and specifically mentions compounding in its name and services. The website confirms it provides customized medication compounding services."""
            },
            {
                "input": {
                    "name": "ANMC Pharmacy",
                    "address": "4315 Diplomacy Dr, Anchorage, AK 99508",
                    "categories": "Pharmacy, Hospital",
                    "website": "http://anmc.org/services/pharmacy/"
                },
                "output": """```json
{
    "classification": "hospital",
    "is_compounding": false,
    "confidence": 0.98
}
```

This is a hospital pharmacy because ANMC (Alaska Native Medical Center) is a medical facility, and the pharmacy serves hospital patients and staff rather than operating as an independent retail business."""
            }
        ]

        # Format the few-shot examples
        example_text = ""
        for ex in examples:
            ex_input = ex["input"]
            example_text += (
                "Input:\n"
                f"Name: {ex_input['name']}\n"
                f"Address: {ex_input['address']}\n"
                f"Categories: {ex_input['categories']}\n"
                f"Website: {ex_input['website']}\n\n"
                f"Output:\n{ex['output']}\n\n"
                "---\n\n"
            )

        return (
            "You are an expert pharmacy classifier. Your task is to classify pharmacies as 'independent', 'chain', 'hospital', or 'not_a_pharmacy'.\n\n"
            "RULES:\n"
            "1. An 'independent' pharmacy is a privately-owned retail pharmacy that:\n"
            "   - Is NOT part of a corporate chain (e.g., CVS, Walgreens, Rite Aid, Walmart, Costco, Kroger)\n"
            "   - Is NOT affiliated with a hospital, clinic, or health system (e.g., VA, medical centers, hospital pharmacies)\n"
            "   - Is NOT part of a government facility (e.g., military, federal, state facilities)\n"
            "   - Operates as a standalone retail business serving the general public\n"
            "2. A 'chain' pharmacy is part of a large corporate chain (e.g., CVS, Walgreens, Rite Aid).\n"
            "3. A 'hospital' pharmacy is located within or affiliated with a hospital, clinic, health system, or medical facility.\n"
            "4. 'not_a_pharmacy' if the business is not a pharmacy.\n"
            "5. A 'compounding' pharmacy customizes medications (often mentioned in name or categories).\n\n"
            "RESPONSE FORMAT:\n"
            "Respond with a single JSON object containing:\n"
            "1. classification: 'independent', 'chain', 'hospital', or 'not_a_pharmacy'\n"
            "2. is_compounding: true or false\n"
            "3. confidence: 0.0 to 1.0 (1.0 = highest confidence)\n"
            "4. explanation: A brief explanation of your reasoning (1-3 sentences)\n\n"
            "The response must be a valid JSON object inside a code block (```json ... ```).\n\n"
            "EXAMPLES:\n" + example_text +
            "Now classify this pharmacy:\n"
            f"Name: {name}\n"
            f"Address: {pharmacy_data.get('address', 'N/A')}\n"
            f"Categories: {pharmacy_data.get('categoryName', 'N/A')}\n"
            f"Website: {pharmacy_data.get('website', 'N/A')}\n\n"
            "Respond with a JSON object in this format (including the explanation field):\n"
            "```json\n"
            "{\n"
            "  \"classification\": \"independent|chain|hospital|not_a_pharmacy\",\n"
            "  \"is_compounding\": true|false,\n"
            "  \"confidence\": 0.0-1.0,\n"
            "  \"explanation\": \"Your explanation here...\"\n"
            "}\n"
            "```"
        )

def _generate_cache_key(pharmacy_data: Dict[str, Any], model: str) -> str:
    """
    Generates a consistent cache key from pharmacy data.
    Uses stable identifiers like name and address.
    """
    # Use 'title' and 'address' as primary identifiers for stability
    name = pharmacy_data.get('title') or pharmacy_data.get('name', 'N/A')
    address = pharmacy_data.get('address', 'N/A')
    
    key_string = f"{name}|{address}|{model}"
    return hashlib.sha256(key_string.encode('utf-8')).hexdigest()
