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
        super().__init__(f"[{self.error_type}] {self.message}")


class PerplexityClient:
    """
    A client for the Perplexity API, designed for pharmacy classification.

    This client handles API requests, rate limiting, and file-based caching
    of classification results.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        rate_limit: int = 20,
        cache_dir: Optional[str] = "data/cache/classification",
        force_reclassification: bool = False
    ):
        """
        Initializes the PerplexityClient.

        Args:
            api_key: The Perplexity API key. If None, it's read from PERPLEXITY_API_KEY env var.
            model_name: The model to use for classification.
            rate_limit: The number of requests per minute to allow.
            cache_dir: The directory to store cache files. Caching is disabled if None.
        """
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        
        # Debug logging to track API key assignment
        logger.debug(f"PerplexityClient constructor - api_key param: {type(api_key)} = {api_key}")
        logger.debug(f"PerplexityClient constructor - os.getenv result: {type(os.getenv('PERPLEXITY_API_KEY'))} = {os.getenv('PERPLEXITY_API_KEY', 'NOT_FOUND')}")
        logger.debug(f"PerplexityClient constructor - self.api_key final: {type(self.api_key)} = {self.api_key}")
        
        if not self.api_key:
            raise ValueError("Perplexity API key not found. Set PERPLEXITY_API_KEY environment variable.")

        config = get_config()
        self.model = model_name or config.get("perplexity_model", "sonar")
        
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.perplexity.ai")
        self.rate_limiter = RateLimiter(rate_limit)
        self.force_reclassification = force_reclassification
        
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"PerplexityClient caching is enabled. Cache directory: {self.cache_dir}")
        else:
            logger.info("PerplexityClient caching is disabled.")

    def classify_pharmacy(self, pharmacy_data: Dict[str, Any], model: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Classifies a single pharmacy using the Perplexity API, with caching.

        Args:
            pharmacy_data: A dictionary containing the pharmacy's details.
            model: The classification model to use.

        Returns:
            A dictionary with the classification results or None if an error occurs.
        """
        model_to_use = model or self.model

        if not self.cache_dir:
            logger.debug("Cache is disabled, calling API directly.")
            return self._make_api_call(pharmacy_data, model_to_use)

        cache_key = _generate_cache_key(pharmacy_data, model_to_use)
        cache_file = self.cache_dir / f"{cache_key}.json"

        # If force_reclassification is False, check cache
        if not self.force_reclassification and cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached_result = json.load(f)
                logger.info(f"CACHE HIT for pharmacy: {pharmacy_data.get('title', 'N/A')}")
                return cached_result
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to read cache file {cache_file}: {e}. Re-fetching.")

        if self.force_reclassification:
            logger.info(f"CACHE IGNORED (force_reclassification=True) for pharmacy: {pharmacy_data.get('title', 'N/A')}. Calling API.")
        else:
            logger.info(f"CACHE MISS for pharmacy: {pharmacy_data.get('title', 'N/A')}. Calling API.")
        
        result = self._make_api_call(pharmacy_data, model_to_use)

        # Write to cache if the API call was successful
        if result:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                with open(cache_file, 'w') as f:
                    json.dump(result, f, indent=4)
                logger.debug(f"SUCCESS: Wrote result to cache file: {cache_file}")
            except IOError as e:
                logger.error(f"Failed to write to cache file {cache_file}: {e}")

        return result    
        
    def _make_api_call(self, pharmacy_data: Dict[str, Any], model: str) -> Optional[Dict[str, Any]]:
        """
        Makes the actual API call to Perplexity.
        """
        pharmacy_name = pharmacy_data.get('title') or pharmacy_data.get('name', 'Unknown')
        logger.info(f"Making Perplexity API call for pharmacy: {pharmacy_name}")
        
        self.rate_limiter.wait()
        prompt = self._generate_prompt(pharmacy_data)
        
        logger.debug(f"Generated prompt for {pharmacy_name}: {prompt[:200]}...")
        
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
