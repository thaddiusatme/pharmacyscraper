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
from src.config import get_config
from src.classification.cache import cache_wrapper

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RateLimiter:
    """A simple rate limiter to control the frequency of API calls."""

    def __init__(self, requests_per_minute: int):
        if requests_per_minute <= 0:
            self.min_interval = 0
        else:
            self.min_interval = 60.0 / requests_per_minute
        self.last_request_time = 0

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
        model: str = "sonar-pro",
        rate_limit: int = 60,  
        max_retries: int = 3,
        temperature: float = 0.1,
        max_tokens: int = 500,
        api_key: Optional[str] = None,
        cache_ttl: int = 300,  
        cache_dir: Optional[Path] = None):
        """Initializes the Perplexity API client.

        Args:
            model: The model to use for classification.
            rate_limit: The number of requests per minute to allow.
            max_retries: The maximum number of retries for a request.
            temperature: The sampling temperature for the model.
            max_tokens: The maximum number of tokens to generate.
            api_key: The Perplexity API key. If not provided, it will be read from
                the PERPLEXITY_API_KEY environment variable.
            cache_ttl: The time-to-live for cache entries in seconds.
            cache_dir: The directory to store cache files.
        """
        config = get_config()
        self.api_key = api_key or config.get("PERPLEXITY_API_KEY")
        if not self.api_key:
            raise ValueError("Perplexity API key not provided or found in environment.")

        self.model = model
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.rate_limiter = RateLimiter(rate_limit)
        self.client = openai.OpenAI(api_key=self.api_key, base_url="https://api.perplexity.ai")

        self.cache_ttl = cache_ttl

        # Initialize cache
        self.cache = None

        logger.info(f"Initialized PerplexityClient with model {self.model} and rate limit {rate_limit} RPM")
    
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
                # Handle sonar-pro format: JSON followed by explanation text
                # Find the first complete JSON object
                content = content.strip()
                if content.startswith('{'):
                    # Find the end of the JSON object by counting braces
                    brace_count = 0
                    json_end = 0
                    for i, char in enumerate(content):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                json_end = i + 1
                                break
                    json_str = content[:json_end]
                else:
                    json_str = content
        
            return json.loads(json_str)
        except (json.JSONDecodeError, AttributeError, IndexError) as e:
            logger.error(f"Failed to parse API response: {e}\nContent: {content}")
            return None

    def _generate_cache_key(self, pharmacy_data: Dict[str, Any], model: str) -> str:
        """Generate a consistent cache key for a pharmacy and model."""
        import json
        import hashlib
        
        # Create a stable string representation of the pharmacy data
        data_str = json.dumps(pharmacy_data, sort_keys=True)
        # Combine with model name and hash
        key_str = f"{data_str}:{model}"
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
                if self.rate_limiter:
                    self.rate_limiter.wait()
                
                # Make the API call
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    **kwargs
                )
                
                return response
                
            except Exception as e:
                last_exception = e
                if attempt == self.max_retries:
                    break
                    
                # Check if this is a rate limit error
                if "rate limit" in str(e).lower() or "too many" in str(e).lower():
                    # Exponential backoff
                    sleep_time = 2 ** attempt
                    time.sleep(sleep_time)
                else:
                    # For other errors, just retry after the base delay
                    time.sleep(1)
        
        # If we get here, all retries failed
        if "rate limit" in str(last_exception).lower() or "too many" in str(last_exception).lower():
            raise RateLimitError(f"Rate limited after {self.max_retries} retries: {last_exception}")
        raise PerplexityAPIError(f"API request failed after {self.max_retries} retries: {last_exception}")

    def _rate_limit_retry(self):
        """Create a retry decorator with rate limiting."""
        return retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=4, max=10),
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
    
    @cache_wrapper
    def classify_pharmacy(self, pharmacy_data: Dict[str, Any], model: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Classify a pharmacy using the Perplexity API."""
        current_model = model or self.model
        cache_key = self._generate_cache_key(pharmacy_data, current_model)

        logger.info(f"Cache miss for pharmacy: {pharmacy_data.get('name', 'N/A')}. Calling API.")
        prompt = self._generate_prompt(pharmacy_data)
        
        try:
            # Use the higher-level helper that wraps message construction so the
            # payload matches the OpenAI-compatible schema expected by the
            # Perplexity endpoint.
            parsed_data = self._call_api(prompt)

            return parsed_data
        except PerplexityAPIError as e:
            logger.error(f"Error classifying pharmacy {pharmacy_data.get('name', 'N/A')}: {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred while classifying {pharmacy_data.get('name', 'N/A')}: {e}")
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
    return client.classify_pharmacy(pharmacy_data, model)

def _generate_cache_key(pharmacy_data: Dict[str, Any], model: str) -> str:
    """
    DEPRECATED: Generate a cache key for the given pharmacy data and model.
    This function is kept for backward compatibility with tests and will be removed.
    """
    import json
    import hashlib

    # Create a stable string representation of the pharmacy data
    data_str = json.dumps(pharmacy_data, sort_keys=True)
    # Combine with model name and hash
    key_str = f"{data_str}:{model}"
    return hashlib.md5(key_str.encode('utf-8')).hexdigest()


__all__ = ['PerplexityClient', '_generate_cache_key', 'PerplexityAPIError', 'RateLimitError']
