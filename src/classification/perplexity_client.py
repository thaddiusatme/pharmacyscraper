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

# Import Cache with absolute import to avoid circular imports
from src.classification.cache import Cache as CacheBase

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Export Cache class for testing
Cache = CacheBase

class PerplexityAPIError(Exception):
    """Base exception for Perplexity API errors."""
    def __init__(self, message, error_type="api_error"):
        self.message = message
        self.error_type = error_type
        super().__init__(self.message)

class RateLimitError(PerplexityAPIError):
    """Raised when the rate limit is exceeded."""
    def __init__(self, message="Rate limit exceeded"):
        super().__init__(message, "rate_limit")

class InvalidRequestError(PerplexityAPIError):
    """Raised when the API request is invalid."""
    def __init__(self, message="Invalid request"):
        super().__init__(message, "invalid_request")

class ResponseParseError(PerplexityAPIError):
    """Raised when the API response cannot be parsed."""
    def __init__(self, message="Failed to parse API response"):
        super().__init__(message, "response_parsing")

class PerplexityClient:
    """
    A client for interacting with the Perplexity API with rate limiting and caching.
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        model: str = "sonar-small-online",
        rate_limit_per_minute: int = 20,
        max_retries: int = 3,
        cache_ttl: int = 300,  # 5 minutes default TTL
        cache_dir: Optional[Union[str, Path]] = None,
        retry_delay: int = 1,
        temperature: float = 0.1,
        max_tokens: int = 500,
        **kwargs  # Accept additional arguments for backward compatibility including rate_limit_ms
    ):
        """
        Initialize the Perplexity client.
        
        Args:
            api_key: Perplexity API key. If not provided, will look for PERPLEXITY_API_KEY env var.
            model: The model to use for completions.
            rate_limit_per_minute: Maximum number of requests per minute.
            max_retries: Maximum number of retries for failed requests.
            cache_ttl: Time-to-live for cache entries in seconds.
            cache_dir: Directory to store cache files. If None, uses a default location.
            retry_delay: Delay between retries in seconds.
            temperature: Temperature for the model.
            max_tokens: Maximum number of tokens for the model.
            **kwargs: Additional arguments for backward compatibility.
        """
        # Handle rate_limit_ms for backward compatibility
        if 'rate_limit_ms' in kwargs and kwargs['rate_limit_ms'] is not None:
            if kwargs['rate_limit_ms'] > 0:
                rate_limit_per_minute = 60000 // kwargs['rate_limit_ms']
            else:
                rate_limit_per_minute = 0  # No rate limiting when rate_limit_ms is 0
        
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        if not self.api_key:
            logger.warning("No API key provided. Set PERPLEXITY_API_KEY environment variable.")
        
        self.model = model
        self.rate_limit_per_minute = rate_limit_per_minute
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize the OpenAI client with Perplexity's API
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://api.perplexity.ai"
        )
        
        # Initialize cache
        cache_kwargs = {
            'name': 'perplexity_cache',
            'ttl': cache_ttl
        }
        if cache_dir:
            cache_kwargs['cache_dir'] = cache_dir
        self.cache = Cache(**cache_kwargs)

        # Rate limiting
        if self.rate_limit_per_minute > 0:
            self.min_interval = 60.0 / rate_limit_per_minute
        else:
            self.min_interval = 0
        
        self.last_request_time = 0
        
        logger.info(f"Initialized PerplexityClient with model {model} and rate limit {rate_limit_per_minute} RPM")
    
    def _parse_response(self, response: Any) -> Optional[Dict[str, Any]]:
        """Parses the response from the Perplexity API."""
        if not response:
            return None

        try:
            # Handle both mocked test responses and real API responses
            if isinstance(response, dict) and 'text' in response:
                content = response['text']
            elif hasattr(response, 'choices') and response.choices:
                content = response.choices[0].message.content
            else:
                logger.error(f"Unexpected response format: {response}")
                return None

            # Extract JSON from markdown code blocks if present
            match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
            if match:
                json_str = match.group(1)
            else:
                json_str = content
            
            return json.loads(json_str)
        except (json.JSONDecodeError, AttributeError, IndexError) as e:
            logger.error(f"Failed to parse API response: {e}\nContent: {content}")
            return None

    def _generate_cache_key(self, pharmacy_data: Dict[str, Any], model: str = None) -> str:
        """Generate a consistent cache key for a pharmacy and model."""
        import json
        import hashlib
        
        # Create a stable string representation of the pharmacy data
        data_str = json.dumps(pharmacy_data, sort_keys=True)
        # Combine with model name and hash
        key_str = f"{data_str}:{model or self.model}"
        return hashlib.md5(key_str.encode('utf-8')).hexdigest()
    
    def _make_request(self, messages, model=None, **kwargs):
        """Make an API request with retry and rate limiting.
        
        Args:
            messages: List of message dictionaries for the chat completion.
            model: The model to use for the request. Uses instance model if None.
            **kwargs: Additional arguments to pass to the API.
            
        Returns:
            The API response.
            
        Raises:
            RateLimitError: If rate limited and retries are exhausted.
            PerplexityAPIError: For other API errors.
        """
        model = model or self.model
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                # Apply rate limiting
                if self.min_interval > 0:
                    current_time = time.time()
                    time_since_last = current_time - self.last_request_time
                    if time_since_last < self.min_interval:
                        time.sleep(self.min_interval - time_since_last)
                
                # Make the API call
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    **kwargs
                )
                
                # Update last request time
                if self.min_interval > 0:
                    self.last_request_time = time.time()
                
                return response
                
            except Exception as e:
                last_exception = e
                if attempt == self.max_retries:
                    break
                    
                # Check if this is a rate limit error
                if "rate limit" in str(e).lower() or "too many" in str(e).lower():
                    # Exponential backoff
                    sleep_time = self.retry_delay * (2 ** attempt)
                    time.sleep(sleep_time)
                else:
                    # For other errors, just retry after the base delay
                    time.sleep(self.retry_delay)
        
        # If we get here, all retries failed
        if "rate limit" in str(last_exception).lower() or "too many" in str(last_exception).lower():
            raise RateLimitError(f"Rate limited after {self.max_retries} retries: {last_exception}")
        raise PerplexityAPIError(f"API request failed after {self.max_retries} retries: {last_exception}")

    def _rate_limit_retry(self):
        """Create a retry decorator with rate limiting."""
        return retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=self.retry_delay, min=4, max=10),
            retry=retry_if_exception_type((
                openai.RateLimitError,
                openai.APITimeoutError,
                openai.APIConnectionError
            )),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True
        )
    
    def _call_api(self, prompt: str) -> Dict[str, Any]:
        """Make an API call to Perplexity with retry logic."""
        @self._rate_limit_retry()
        def _make_api_call():
            self._handle_rate_limit()
            response = self._make_request(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            return self._parse_response(response)
            
        try:
            return _make_api_call()
        except Exception as e:
            logger.error(f"Error calling Perplexity API: {e}")
            raise
    
    def _handle_rate_limit(self):
        """Handle rate limiting by sleeping if needed."""
        current_time = time.time()
        
        # Check if we need to sleep
        if current_time - self.last_request_time < self.min_interval:
            sleep_time = self.min_interval - (current_time - self.last_request_time)
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s between requests")
            time.sleep(sleep_time)
        
        # Update last request time
        self.last_request_time = time.time()
    
    def classify_pharmacy(self, pharmacy_data: Dict[str, Any], model: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Classify a pharmacy using the Perplexity API."""
        current_model = model or self.model
        cache_key = self._generate_cache_key(pharmacy_data, current_model)

        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            logger.info(f"Cache hit for pharmacy: {pharmacy_data.get('name')}")
            return cached_result

        logger.info(f"Cache miss for pharmacy: {pharmacy_data.get('name')}. Calling API.")
        prompt = self._generate_prompt(pharmacy_data)
        
        try:
            response = self._make_request(prompt, current_model)
            parsed_data = self._parse_response(response)

            if parsed_data:
                self.cache.set(cache_key, parsed_data)
            
            return parsed_data
        except PerplexityAPIError as e:
            logger.error(f"Error classifying pharmacy {pharmacy_data.get('name')}: {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred while classifying {pharmacy_data.get('name')}: {e}")
            return None

    def classify_pharmacies_batch(self, pharmacies_data: List[Dict[str, Any]], model: Optional[str] = None) -> List[Optional[Dict[str, Any]]]:
        """Classifies a batch of pharmacies.

        Args:
            pharmacies_data: A list of pharmacy data dictionaries.
            model: The model to use for classification.

        Returns:
            A list of classification results.
        """        
        return [self.classify_pharmacy(pharmacy_data, model) for pharmacy_data in pharmacies_data]

    def _generate_prompt(self, pharmacy_data: Dict[str, Any]) -> str:
        """Generate a prompt for pharmacy classification."""
        return (
            f"Classify the following pharmacy:\n"
            f"Name: {pharmacy_data.get('name', 'N/A')}\n"
            f"Address: {pharmacy_data.get('address', 'N/A')}\n"
            f"Phone: {pharmacy_data.get('phone', 'N/A')}\n"
            f"Website: {pharmacy_data.get('website', 'N/A')}\n\n"
            "Is this an independent pharmacy? Respond with a JSON object containing 'is_pharmacy' (boolean), "
            "'is_compound_pharmacy' (boolean), and 'confidence' (float between 0 and 1)."
        )

# Module-level function for cache key generation
def _generate_cache_key(pharmacy_data: Dict[str, Any], model: str) -> str:
    """Generate a cache key for the given pharmacy data and model.
    
    Args:
        pharmacy_data: Dictionary containing pharmacy information.
        model: The model name to include in the cache key.
        
    Returns:
        str: A unique cache key.
    """
    import json
    import hashlib
    
    # Create a stable string representation of the pharmacy data
    data_str = json.dumps(pharmacy_data, sort_keys=True)
    # Combine with model name and hash
    key_str = f"{data_str}:{model}"
    return hashlib.md5(key_str.encode('utf-8')).hexdigest()

# Helper function for simple usage
def classify_pharmacy(pharmacy_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    Classify a single pharmacy using the Perplexity API.
    
    Args:
        pharmacy_data: Dictionary containing pharmacy information.
        **kwargs: Additional arguments to pass to PerplexityClient.
        
    Returns:
        Dictionary with classification results.
    """
    # Remove cache_dir from kwargs if it exists to avoid passing it twice
    kwargs.pop('cache_dir', None)
    client = PerplexityClient(**kwargs)
    return client.classify_pharmacy(pharmacy_data)

# Export for testing
__all__ = ['PerplexityClient', '_generate_cache_key', 'PerplexityAPIError', 'RateLimitError', 'Cache']
