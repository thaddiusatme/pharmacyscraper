"""
Perplexity API client for pharmacy classification.

This module provides a client for interacting with the Perplexity API
using the OpenAI-compatible interface.
"""
import os
import json
import time
import logging
import random
import re
from pathlib import Path
import hashlib
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
from pharmacy_scraper.utils import Cache
from .models import ClassificationResult, PharmacyData, ClassificationSource

# Configure logging
logger = logging.getLogger(__name__)

__all__ = [
    "PerplexityClient",
    "PerplexityAPIError",
    "RateLimitError",
    "InvalidRequestError",
    "ResponseParseError",
    "RateLimiter",
    "_generate_cache_key",
]


# ---------------------------------------------------------------------------
# Public helper – exposed for the test-suite
# ---------------------------------------------------------------------------

def _generate_cache_key(pharmacy_data: Dict[str, Any], model: Optional[str] = None) -> str:
    """Generate a deterministic hash for *(pharmacy_data, model)*.

    The unit-tests import this symbol directly so keep the signature stable.
    """
    # Convert dataclass instances to raw dicts if needed
    if hasattr(pharmacy_data, "to_dict"):
        pharmacy_data = pharmacy_data.to_dict()

    data_str = json.dumps(pharmacy_data, sort_keys=True)
    if model:
        data_str += f"|{model}"

    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


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
        
        # Calculate how long to wait before the next request is allowed
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            time.sleep(sleep_time)
            
        self.last_request_time = time.time()


class PerplexityAPIError(Exception):
    """Base exception for Perplexity API errors."""
    
    def __init__(self, message, error_type="api_error"):
        self.message = message
        self.error_type = error_type
        super().__init__(f"[{self.error_type}] {self.message}")


# Additional exceptions for test compatibility
class RateLimitError(PerplexityAPIError):
    """Exception for rate limit exceeded."""
    pass


class InvalidRequestError(PerplexityAPIError):
    """Exception for invalid request to Perplexity API."""
    pass


class ResponseParseError(PerplexityAPIError):
    """Exception for errors parsing Perplexity API response."""
    pass


class PerplexityClient:
    """Client for the Perplexity API used for pharmacy classification.
    
    This client handles API calls, caching, and response parsing to classify pharmacies.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "pplx-7b-chat",
        temperature: float = 0.1,
        cache_enabled: bool = True,
        cache_dir: Optional[str] = None,
        cache_ttl_seconds: Optional[int] = None,
        force_reclassification: bool = False,
        system_prompt: str = "You are a pharmacy classification expert. Your task is to classify pharmacies based on their details."
    ):
        """Initialize the PerplexityClient.
        
        Args:
            api_key: The API key for the Perplexity API. If None, it will be read from the environment.
            model_name: The model to use for classification.
            temperature: The temperature parameter for the API calls.
            cache_enabled: Whether to enable caching of classification results.
            force_reclassification: Whether to force reclassification even if a cached result exists.
            system_prompt: The system prompt to use for classification.
        """
        # Use provided API key or get from environment/config
        self.api_key = api_key or os.environ.get("PERPLEXITY_API_KEY") or get_config().get("perplexity_api_key")
        if not self.api_key:
            raise ValueError("No Perplexity API key provided. Set PERPLEXITY_API_KEY environment variable or pass it explicitly.")
            
        # Initialize API client
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.perplexity.ai")
        
        # Configuration
        self.model_name = model_name
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.force_reclassification = force_reclassification
        
        # Setup cache
        if cache_enabled:
            cache_directory = cache_dir
            # If no custom cache directory is provided, create one based on the perplexity namespace
            if not cache_directory:
                from pathlib import Path
                cache_directory = Path.home() / ".cache" / "pharmacy_scraper" / "perplexity"
            self.cache = Cache(cache_dir=cache_directory, ttl=cache_ttl_seconds)
        else:
            self.cache = None
        
        # Rate limiting
        self.rate_limiter = RateLimiter(requests_per_minute=10)  # Default rate limit
    
    def classify_pharmacy(
        self,
        pharmacy_data: Union[Dict[str, Any], "PharmacyData"],
        model: Optional[str] = None,
    ) -> "ClassificationResult":
        """Classify a single pharmacy, handling caching and API calls."""
        if not isinstance(pharmacy_data, PharmacyData):
            try:
                pharmacy_data = PharmacyData.from_dict(pharmacy_data)
            except (AttributeError, TypeError, ValueError) as e:
                # Wrap input validation errors in PerplexityAPIError
                raise PerplexityAPIError(f"Invalid input data: {e}", error_type="invalid_input")

        # If force_reclassification is False, check the cache first
        if not self.force_reclassification:
            cache_key = _generate_cache_key(pharmacy_data.to_dict(), self.model_name)
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                # The cached result's pharmacy_data might be a dict, so ensure it's a PharmacyData instance
                if isinstance(cached_result.pharmacy_data, dict):
                    p_data = PharmacyData.from_dict(cached_result.pharmacy_data)
                else:
                    p_data = cached_result.pharmacy_data

                return ClassificationResult(
                    classification=cached_result.classification,
                    is_chain=cached_result.is_chain,
                    is_compounding=cached_result.is_compounding,
                    confidence=cached_result.confidence,
                    explanation=cached_result.explanation,
                    source=ClassificationSource.CACHE, # Explicitly set source
                    model=cached_result.model,
                    pharmacy_data=p_data,
                    error=cached_result.error
                )

        # If not in cache or force_reclassification is True, call the API
        try:
            response = self._call_api_with_retries(pharmacy_data)
            result = self._parse_response(response, pharmacy_data)

            # Save to cache if caching is enabled
            if self.cache:
                cache_key = _generate_cache_key(pharmacy_data.to_dict(), self.model_name)
                self._save_to_cache(cache_key, result)
            return result

        except Exception as e:
            # Handle API-related errors
            logger.error(f"API error occurred: {e}")
            raise PerplexityAPIError(f"API error: {e}", error_type="api_error") from e

    def _make_api_call(self, pharmacy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make an API call to classify a pharmacy.
        
        This method is used by the tests to mock API calls.
        
        Args:
            pharmacy_data: Dictionary containing pharmacy data.
            
        Returns:
            Dictionary containing the classification result.
        """
        # Ensure pharmacy_data is a PharmacyData instance
        if not isinstance(pharmacy_data, PharmacyData):
            pharmacy_data = PharmacyData.from_dict(pharmacy_data)
            
        # This method is primarily mocked in tests
        # For compatibility with the test data, we include both explanation and reason keys
        response = self._call_api_with_retries(pharmacy_data)
        result = self._parse_response(response, pharmacy_data)
        result_dict = result.to_dict()
        
        # Add reason key for backward compatibility with tests
        if 'explanation' in result_dict and 'reason' not in result_dict:
            result_dict['reason'] = result_dict['explanation']
            
        return result_dict
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APIStatusError)),
        before_sleep=before_sleep_log(logger, logging.INFO)
    )
    def _call_api_with_retries(self, pharmacy_data: PharmacyData) -> Any:
        """Internal method to call the Perplexity API with retry logic."""
        self.rate_limiter.wait()
        prompt = self._create_prompt(pharmacy_data)
        
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
        )

    def _get_from_cache(self, key: str) -> Optional[ClassificationResult]:
        """Retrieve a classification result from the cache."""
        if not self.cache:
            return None
           
        try: 
            cached_data = self.cache.get(key)
            if cached_data and isinstance(cached_data, dict):
                try:
                    # Reconstruct PharmacyData if it exists in cache
                    pharmacy_data_dict = cached_data.get('pharmacy_data')
                    pharmacy_data = PharmacyData.from_dict(pharmacy_data_dict) if pharmacy_data_dict else None

                    return ClassificationResult(
                        classification=cached_data.get('classification'),
                        is_chain=cached_data.get('is_chain'),
                        is_compounding=cached_data.get('is_compounding'),
                        confidence=cached_data.get('confidence'),
                        explanation=cached_data.get('explanation'),
                        model=cached_data.get('model'),
                        pharmacy_data=pharmacy_data,
                        source=ClassificationSource(cached_data.get('source', 'cache')),
                        error=cached_data.get('error')
                    )
                except (KeyError, TypeError, ValueError) as e:
                    logger.warning(f"Failed to load cached result for key {key} due to invalid data: {e}")
                    try:
                        self.cache.delete(key) # Corrupted cache entry
                    except (IOError, OSError) as delete_error:
                        logger.warning(f"Failed to delete corrupted cache entry: {delete_error}")
                    return None
            elif cached_data:
                logger.warning(f"Invalid cache format for key {key}: expected dict, got {type(cached_data)}")
            return None
        except (IOError, OSError) as e:
            # Handle any IO errors that might occur when reading from cache
            logger.warning(f"Cache read error for key {key}: {e}")
            return None
            
        return None

    def _save_to_cache(self, key: str, result: ClassificationResult):
        """Save a classification result to the cache.
        
        This method will catch and log any IOErrors that occur during cache writing,
        ensuring that cache errors don't propagate to the calling code.
        
        Args:
            key: The cache key to use
            result: The ClassificationResult to cache
        """
        if self.cache:
            try:
                self.cache.set(key, result.to_dict())
            except IOError as e:
                # Log the error but don't crash the application
                logger.error("Cache write error: %s", str(e))

    def _parse_response(self, response: Any, pharmacy_data: PharmacyData) -> ClassificationResult:
        """Parse the API response and create a ClassificationResult."""
        try:
            content = response.choices[0].message.content
            # Basic cleaning to remove markdown fences
            if content.startswith('```json'):
                content = content.strip('```json').strip('`').strip()
            
            parsed_content = json.loads(content)

            # Basic validation of parsed content
            if 'classification' not in parsed_content or 'is_chain' not in parsed_content:
                raise ResponseParseError(
                    f"Missing required keys ('classification', 'is_chain') in response: {parsed_content}"
                )

            return ClassificationResult(
                classification=parsed_content.get('classification'),
                is_chain=parsed_content.get('is_chain'),
                is_compounding=parsed_content.get('is_compounding', False),
                confidence=parsed_content.get('confidence', 0.0),
                explanation=parsed_content.get('explanation', ''),
                source=ClassificationSource.PERPLEXITY,
                model=self.model_name,
                pharmacy_data=pharmacy_data
            )
        except (json.JSONDecodeError, IndexError, KeyError, AttributeError) as e:
            logger.error(f"Failed to parse Perplexity API response: {e}")
            raise ResponseParseError(f"Invalid response format: {e}") from e

    def _create_prompt(self, pharmacy_data: PharmacyData) -> str:
        """Create the prompt for the Perplexity API."""
        # Using double curly braces to escape them in f-strings
        return f"""Please classify the following pharmacy based on its details.
Name: {pharmacy_data.name}
Address: {pharmacy_data.address}
Categories: {pharmacy_data.categories}
Website: {pharmacy_data.website}

Please respond with a JSON object containing the classification, is_chain, is_compounding, confidence, and explanation.
```json
{{
    "classification": "independent|chain|hospital|not_a_pharmacy",
    "is_chain": true|false,
    "is_compounding": true|false,
    "confidence": 0.0-1.0,
    "explanation": "Your explanation here..."
}}
```"""
