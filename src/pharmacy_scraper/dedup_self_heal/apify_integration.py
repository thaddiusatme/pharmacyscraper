"""
Apify integration for scraping pharmacy data.

This module provides an interface to the Apify Google Maps Scraper actor
for collecting pharmacy data with credit management and rate limiting.
"""
import os
import time
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, retry_if_exception
from apify_client import ApifyClient

# Import credit tracker
from ..utils.api_usage_tracker import credit_tracker, CreditLimitExceededError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    'max_results': 10,  # Reduced from 25 to be more conservative
    'max_retries': 3,
    'initial_delay': 1,
    'max_delay': 10,
    'requests_per_minute': 10,  # Rate limiting
    'estimated_cost_per_call': 0.1,  # Estimated cost in credits per API call
}

class ApifyScraperError(Exception):
    """Base exception for Apify scraper errors."""
    pass

class RateLimitExceededError(Exception):
    """Raised when rate limits are exceeded."""
    pass

class ApifyPharmacyScraper:
    """Client for scraping pharmacy data using Apify's Google Maps Scraper with credit management."""
    
    def __init__(self, api_token: str = None, config: Dict[str, Any] = None):
        """Initialize the Apify pharmacy scraper with credit tracking.
        
        Args:
            api_token: Apify API token. If not provided, will look for APIFY_TOKEN
                     or APIFY_API_TOKEN environment variables.
            config: Configuration dictionary. See DEFAULT_CONFIG for available options.
        """
        self.api_token = api_token or os.getenv('APIFY_TOKEN') or os.getenv('APIFY_API_TOKEN')
        if not self.api_token:
            raise ValueError(
                "Apify API token is required. "
                "Either pass it to the constructor or set APIFY_TOKEN/APIFY_API_TOKEN environment variable."
            )
            
        self.client = ApifyClient(self.api_token)
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 60.0 / self.config['requests_per_minute']
        
        logger.info("ApifyPharmacyScraper initialized with rate limiting and credit tracking")
    
    def _check_credits_available(self, operation: str = "api_call") -> bool:
        """Check if credits are available for an operation.
        
        Args:
            operation: Description of the operation for logging
            
        Returns:
            bool: True if credits are available, False otherwise
            
        Raises:
            CreditLimitExceededError: If credit limit would be exceeded
        """
        if not credit_tracker.check_credit_available(self.config['estimated_cost_per_call']):
            raise CreditLimitExceededError(
                f"Insufficient credits for {operation}. "
                f"Used {credit_tracker.usage_data['total_used']:.2f}/{credit_tracker.budget:.2f} credits"
            )
        return True
    
    def _rate_limit(self):
        """Enforce rate limiting between API calls."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=(
            retry_if_exception_type(Exception) |
            retry_if_exception(lambda e: isinstance(e, CreditLimitExceededError) and \
                              credit_tracker.get_usage_summary()['remaining'] > 0)
        ),
        reraise=True
    )
    def _run_actor(self, state: str, city: str, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Run the Apify actor to scrape pharmacy data with credit and rate limiting."""
        try:
            # Check credits before making the call
            self._check_credits_available(f"Apify search in {city}, {state}")
            
            # Enforce rate limiting
            self._rate_limit()
            
            # Prepare the actor input
            run_input = {
                "searchStringsArray": [query],
                "maxCrawledPlacesPerSearch": max_results,
                "includeWebResults": False,
                "language": "en",
                "maxImages": 0,
                "maxReviews": 0,
                "scraperLocation": "US",
            }
            
            logger.info(f"Starting Apify actor for {query} (max_results: {max_results})")
            
            # Run the actor and wait for it to finish
            run = self.client.actor("nwua9Gu5YrADL7ZDj").call(run_input=run_input)
            
            # Record the API usage
            credit_tracker.record_usage(
                self.config['estimated_cost_per_call'],
                f"Apify search: {query}"
            )
            
            # Fetch the results
            items = list(self.client.dataset(run["defaultDatasetId"]).iterate_items())
            logger.info(f"Retrieved {len(items)} results for {query}")
            
            return items
            
        except Exception as e:
            if "ApifyApiError" in str(type(e).__name__) or "ApiError" in str(type(e).__name__):
                logger.error(f"Apify API error: {str(e)}")
            else:
                logger.error(f"Unexpected error in _run_actor: {str(e)}")
            raise ApifyScraperError(f"Failed to run Apify actor: {str(e)}")
    
    def scrape_pharmacies(self, state: str, city: str = None, query: str = None, 
                         max_results: int = None) -> List[Dict[str, Any]]:
        """Scrape pharmacies with credit and rate limiting.
        
        Args:
            state: Two-letter state code
            city: Optional city name
            query: Custom search query
            max_results: Maximum results to return
            
        Returns:
            List of pharmacy records
        """
        try:
            # Apply limits
            max_results = min(max_results or self.config['max_results'], self.config['max_results'])
            query = query or f"pharmacy in {city + ', ' if city else ''}{state}"
            
            # Get results
            results = self._run_actor(state, city or "", query, max_results)
            
            # Standardize and return
            return [self._standardize_pharmacy(item, state, city) for item in results]
            
        except CreditLimitExceededError:
            logger.warning("Credit limit reached, returning empty results")
            return []
        except Exception as e:
            logger.error(f"Error in scrape_pharmacies: {str(e)}")
            return []
    
    @staticmethod
    def _standardize_pharmacy(data: Dict[str, Any], state: str, city: Optional[str]) -> Dict[str, Any]:
        """Standardize pharmacy data format."""
        address_parts = data.get('address', '').split(',')
        zip_code = address_parts[-1].strip() if len(address_parts) > 1 else ''
        
        return {
            'name': data.get('title', '').strip(),
            'address': data.get('address', '').strip(),
            'city': city or address_parts[-2].strip() if len(address_parts) > 1 else '',
            'state': state,
            'zip': zip_code,
            'phone': data.get('phone', '').strip(),
            'latitude': data.get('location', {}).get('lat'),
            'longitude': data.get('location', {}).get('lng'),
            'website': data.get('website', '').strip(),
            'source': 'apify',
            'scraped_at': data.get('scrapedAt', datetime.utcnow().isoformat()),
            'raw_data': data
        }

def get_apify_client(api_token: str = None) -> ApifyPharmacyScraper:
    """Get a configured ApifyPharmacyScraper instance with credit tracking."""
    return ApifyPharmacyScraper(api_token=api_token)