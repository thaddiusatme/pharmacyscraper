"""
Deduplication and Self-Healing module for pharmacy data.

This module provides functionality to deduplicate pharmacy records and
self-heal by finding additional pharmacies for under-filled states.
"""

from .dedup import (
    group_pharmacies_by_state,
    remove_duplicates,
    identify_underfilled_states,
    self_heal_state,
    merge_new_pharmacies,
    process_pharmacies,
    TARGET_PHARMACIES_PER_STATE,
    DEFAULT_MIN_REQUIRED
)

from .apify_integration import (
    get_apify_client as get_apify_scraper,
    ApifyPharmacyScraper
)

def scrape_pharmacies(state: str, city: str = None, query: str = None, max_results: int = 10):
    """
    Scrape pharmacies using Apify's Google Maps Scraper.
    
    Args:
        state: Two-letter state code
        city: Optional city name
        query: Search query (defaults to 'pharmacy')
        max_results: Maximum number of results to return
        
    Returns:
        DataFrame containing scraped pharmacy data
    """
    scraper = ApifyPharmacyScraper()
    return scraper.scrape_pharmacies(
        state=state,
        city=city,
        query=query or 'pharmacy',
        max_results=max_results
    )

__version__ = "0.1.0"

__all__ = [
    'group_pharmacies_by_state',
    'remove_duplicates',
    'identify_underfilled_states',
    'self_heal_state',
    'merge_new_pharmacies',
    'process_pharmacies',
    'get_apify_scraper',
    'scrape_pharmacies',
    'ApifyPharmacyScraper',
    'TARGET_PHARMACIES_PER_STATE',
    'DEFAULT_MIN_REQUIRED'
]
