"""
Perplexity API client for pharmacy classification.

This module provides a client for interacting with the Perplexity API
using the OpenAI-compatible interface.
"""
import os
import json
import time
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path

import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

# Configure logging
logger = logging.getLogger(__name__)

class PerplexityAPIError(Exception):
    """Base exception for Perplexity API errors."""
    def __init__(self, message: str, error_type: str = "api_error"):
        self.message = message
        self.error_type = error_type
        super().__init__(self.message)

class RateLimitError(PerplexityAPIError):
    """Raised when rate limit is exceeded."""
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message, "rate_limit")

class ResponseParsingError(PerplexityAPIError):
    """Raised when there's an error parsing the API response."""
    def __init__(self, message: str = "Failed to parse API response"):
        super().__init__(message, "response_parsing")

class PerplexityClient:
    """Client for interacting with the Perplexity API."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "sonar-medium-online",
        base_url: str = "https://api.perplexity.ai",
        max_retries: int = 3,
        timeout: int = 30,
        max_requests_per_minute: int = 100
    ):
        """Initialize the Perplexity client.
        
        Args:
            api_key: Perplexity API key. If not provided, will use PERPLEXITY_API_KEY env var.
            model: Model to use for completions.
            base_url: Base URL for the API.
            max_retries: Maximum number of retries for failed requests.
            timeout: Request timeout in seconds.
            max_requests_per_minute: Maximum number of requests per minute.
        """
        self.api_key = api_key or os.getenv('PERPLEXITY_API_KEY')
        if not self.api_key:
            raise ValueError("API key must be provided or set in PERPLEXITY_API_KEY environment variable")
            
        self.model = model or os.getenv('PERPLEXITY_MODEL', 'sonar-medium-online')
        self.base_url = base_url or os.getenv('PERPLEXITY_BASE_URL', 'https://api.perplexity.ai')
        self.max_retries = max_retries or int(os.getenv('PERPLEXITY_MAX_RETRIES', 3))
        self.timeout = timeout or int(os.getenv('PERPLEXITY_TIMEOUT', 30))
        self.max_requests_per_minute = max_requests_per_minute or int(os.getenv('PERPLEXITY_MAX_REQUESTS_PER_MINUTE', 100))
        
        # Track rate limiting
        self.requests_this_minute = 0
        self.last_request_time = 0
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )
    
    def _generate_prompt(self, pharmacy_data: Dict[str, Any]) -> str:
        """Generate a prompt for pharmacy classification.
        
        Args:
            pharmacy_data: Dictionary containing pharmacy information.
            
        Returns:
            Formatted prompt string.
        """
        prompt = (
            "Determine if the following pharmacy is part of a major chain or an independent pharmacy. "
            "Consider the name, address, and any other provided information.\n\n"
            f"Pharmacy Name: {pharmacy_data.get('name', 'N/A')}\n"
            f"Address: {pharmacy_data.get('address', 'N/A')}\n"
            f"Phone: {pharmacy_data.get('phone', 'N/A')}\n\n"
            "Return a JSON object with the following structure:\n"
            "{\n"
            "  \"is_chain\": boolean,  // true if part of a major chain, false if independent\n"
            "  \"confidence\": float,  // confidence score between 0 and 1\n"
            "  \"reason\": string      // brief explanation of the classification\n"
            "}"
        )
        return prompt
    
    def _handle_rate_limit(self):
        """Handle rate limiting by sleeping if needed."""
        current_time = time.time()
        
        # Reset counter if we're in a new minute
        if current_time - self.last_request_time >= 60:
            self.requests_this_minute = 0
            self.last_request_time = current_time
        
        # Sleep if we're approaching the rate limit
        if self.requests_this_minute >= self.max_requests_per_minute:
            sleep_time = 60 - (current_time - self.last_request_time) + 1
            logger.warning(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
            self.requests_this_minute = 0
            self.last_request_time = time.time()
        
        self.requests_this_minute += 1
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((RateLimitError, openai.APIError, openai.APITimeoutError)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def _make_request(self, prompt: str) -> Dict[str, Any]:
        """Make a request to the Perplexity API.
        
        Args:
            prompt: The prompt to send to the API.
            
        Returns:
            The parsed API response.
            
        Raises:
            RateLimitError: If rate limit is exceeded.
            PerplexityAPIError: For other API errors.
        """
        self._handle_rate_limit()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that classifies pharmacies as either independent or part of a chain."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            # Parse response
            content = response.choices[0].message.content
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse API response: {e}")
                raise ResponseParsingError(f"Failed to parse API response: {e}") from e
                
        except openai.RateLimitError as e:
            logger.warning("Rate limit exceeded")
            raise RateLimitError("Rate limit exceeded") from e
            
        except openai.APIError as e:
            logger.error(f"API request failed: {e}")
            raise PerplexityAPIError(f"API request failed: {e}", "api_error") from e
        
        except ResponseParsingError:
            raise  # Re-raise ResponseParsingError directly
            
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise PerplexityAPIError(f"Unexpected error: {e}", "unknown") from e
    
    def classify_pharmacy(self, pharmacy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Classify a pharmacy as independent or part of a chain.
        
        Args:
            pharmacy_data: Dictionary containing pharmacy information.
            
        Returns:
            Dictionary with classification results.
            
        Raises:
            PerplexityAPIError: If classification fails after retries.
        """
        try:
            prompt = self._generate_prompt(pharmacy_data)
            response = self._make_request(prompt)
            
            # Validate response structure
            if not all(k in response for k in ["is_chain", "confidence", "reason"]):
                raise PerplexityAPIError(
                    "Invalid response format from API",
                    error_type="invalid_format"
                )
                
            return {
                "is_chain": bool(response["is_chain"]),
                "confidence": float(response["confidence"]),
                "reason": str(response["reason"])
            }
            
        except PerplexityAPIError:
            raise  # Re-raise our custom errors
        except Exception as e:
            logger.error(f"Failed to classify pharmacy: {e}")
            raise PerplexityAPIError(
                f"Classification failed: {e}",
                error_type="classification_failed"
            ) from e


def get_client() -> PerplexityClient:
    """Get a configured PerplexityClient instance."""
    return PerplexityClient()


def classify_pharmacy(pharmacy_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Classify a pharmacy as independent or part of a chain.
    
    This is a convenience function that creates a new client instance.
    
    Args:
        pharmacy_data: Dictionary containing pharmacy information.
        **kwargs: Additional arguments to pass to PerplexityClient.
        
    Returns:
        Dictionary with classification results.
    """
    # Remove cache_dir from kwargs if present (handled by cache layer)
    kwargs.pop('cache_dir', None)
    client = PerplexityClient(**kwargs)
    return client.classify_pharmacy(pharmacy_data)
