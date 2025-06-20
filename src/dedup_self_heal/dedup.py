"""
Deduplication and Self-Healing module for pharmacy data.

This module handles:
1. Grouping pharmacies by state
2. Removing duplicate pharmacy entries
3. Identifying under-filled states
4. Self-healing by finding additional pharmacies for under-filled states
5. Merging new pharmacy data with existing data
"""
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import logging
import os
from pathlib import Path

# Import Apify integration
from .apify_integration import ApifyPharmacyScraper, ApifyScraperError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
TARGET_PHARMACIES_PER_STATE = 25
DEFAULT_MIN_REQUIRED = 10  # Minimum pharmacies before considering a state under-filled

# Initialize Apify scraper (lazy-loaded)
_apify_scraper = None

def get_apify_scraper() -> ApifyPharmacyScraper:
    """Get a configured ApifyPharmacyScraper instance.
    
    Returns:
        Configured ApifyPharmacyScraper instance
    """
    global _apify_scraper
    if _apify_scraper is None:
        _apify_scraper = ApifyPharmacyScraper()
    return _apify_scraper

def group_pharmacies_by_state(pharmacies: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Group pharmacies by their state.
    
    Args:
        pharmacies: DataFrame containing pharmacy data with a 'state' column
        
    Returns:
        Dictionary mapping state codes to DataFrames of pharmacies in that state
    """
    if 'state' not in pharmacies.columns:
        raise ValueError("Input DataFrame must contain a 'state' column")
        
    return {state: group for state, group in pharmacies.groupby('state')}

def remove_duplicates(pharmacies: pd.DataFrame, 
                    subset: List[str] = None,
                    keep: str = 'first') -> pd.DataFrame:
    """
    Remove duplicate pharmacy entries.
    
    Args:
        pharmacies: DataFrame containing pharmacy data
        subset: List of column names to consider for identifying duplicates.
               If None, all columns are used.
        keep: Which duplicates to keep. 'first' keeps the first occurrence, 
              'last' keeps the last, False keeps none.
              
    Returns:
        DataFrame with duplicates removed
    """
    if subset is None:
        # Default to using these columns for identifying duplicates
        subset = ['name', 'address', 'city', 'state', 'zip']
    
    # Only keep columns that exist in the DataFrame
    subset = [col for col in subset if col in pharmacies.columns]
    
    return pharmacies.drop_duplicates(subset=subset, keep=keep)

def identify_underfilled_states(grouped_pharmacies: Dict[str, pd.DataFrame],
                              min_required: int = DEFAULT_MIN_REQUIRED) -> Dict[str, int]:
    """
    Identify states that don't have enough pharmacies.
    
    Args:
        grouped_pharmacies: Dictionary of {state: DataFrame of pharmacies}
        min_required: Minimum number of pharmacies required per state
        
    Returns:
        Dictionary mapping state codes to number of additional pharmacies needed
    """
    underfilled = {}
    
    for state, df in grouped_pharmacies.items():
        count = len(df)
        if count < min_required:
            underfilled[state] = min_required - count
            
    return underfilled

def scrape_pharmacies(state: str, count: int, city: str = None) -> pd.DataFrame:
    """
    Scrape additional pharmacies for a given state.
    
    This function uses the Apify Google Maps Scraper to find more pharmacies
    in the specified state.
    
    Args:
        state: Two-letter state code
        count: Number of pharmacies to try to find
        city: Optional city to narrow down the search
        
    Returns:
        DataFrame of newly found pharmacies
    """
    try:
        scraper = get_apify_scraper()
        
        # If city is provided, use it in the search query
        query = f"pharmacy in {city}, {state}" if city else f"pharmacy in {state}"
        
        # Scrape pharmacies
        results = scraper.scrape_pharmacies(
            state=state,
            city=city or "",  # Use empty string if city is None
            query=query,
            max_results=count * 2  # Scrape extra to account for potential duplicates
        )
        
        if not results:
            logger.warning(f"No pharmacies found for {query}")
            return pd.DataFrame()
            
        # Convert to DataFrame and add metadata
        df = pd.DataFrame(results)
        df['scraped_at'] = pd.Timestamp.now()
        
        return df
        
    except ApifyScraperError as e:
        logger.error(f"Failed to scrape pharmacies for {state}: {str(e)}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Unexpected error while scraping pharmacies: {str(e)}")
        return pd.DataFrame()

def self_heal_state(state: str, 
                   existing_pharmacies: pd.DataFrame,
                   needed: int,
                   max_scrape: int = 50) -> pd.DataFrame:
    """
    Find additional pharmacies for an under-filled state.
    
    Args:
        state: Two-letter state code
        existing_pharmacies: DataFrame of existing pharmacies in this state
        needed: Number of additional pharmacies needed
        max_scrape: Maximum number of pharmacies to scrape in one call
        
    Returns:
        DataFrame containing both existing and new pharmacies
    """
    # Make a copy to avoid modifying the original
    result = existing_pharmacies.copy()
    
    if needed <= 0:
        return result
        
    try:
        # Try to get the most common city in the existing data
        city = None
        if not existing_pharmacies.empty and 'city' in existing_pharmacies.columns:
            city_counts = existing_pharmacies['city'].value_counts()
            if not city_counts.empty:
                city = city_counts.idxmax()
        
        # Scrape new pharmacies
        logger.info(f"Scraping up to {needed} new pharmacies for {state} (city: {city or 'any'})")
        new_pharmacies = scrape_pharmacies(state, needed, city)
        
        if not new_pharmacies.empty:
            # Remove any duplicates with existing pharmacies
            combined = pd.concat([result, new_pharmacies], ignore_index=True)
            result = remove_duplicates(combined)
            
            # Only keep the number we need
            if len(result) > len(existing_pharmacies) + needed:
                new_count = min(needed, len(result) - len(existing_pharmacies))
                logger.info(f"Found {new_count} new unique pharmacies for {state}")
                
                # Keep all existing and add only the needed new ones
                existing_indices = set(existing_pharmacies.index)
                new_entries = result[~result.index.isin(existing_indices)].head(needed)
                result = pd.concat([existing_pharmacies, new_entries])
    
    except Exception as e:
        logger.error(f"Error in self_heal_state for {state}: {str(e)}")
    
    return result

def merge_new_pharmacies(existing: pd.DataFrame, 
                        new: pd.DataFrame) -> pd.DataFrame:
    """
    Merge new pharmacies with existing ones, removing duplicates.
    
    Args:
        existing: DataFrame of existing pharmacies
        new: DataFrame of new pharmacies to add
        
    Returns:
        Combined DataFrame with duplicates removed
    """
    if existing.empty:
        return new
    if new.empty:
        return existing
        
    combined = pd.concat([existing, new], ignore_index=True)
    return remove_duplicates(combined)

def process_pharmacies(pharmacies: pd.DataFrame,
                      target_per_state: int = TARGET_PHARMACIES_PER_STATE) -> Dict[str, pd.DataFrame]:
    """
    Process pharmacy data to ensure sufficient coverage across all states.
    
    Args:
        pharmacies: DataFrame of pharmacy data
        target_per_state: Target number of pharmacies per state
        
    Returns:
        Dictionary mapping state codes to processed DataFrames of pharmacies
    """
    # Remove duplicates
    unique_pharmacies = remove_duplicates(pharmacies)
    
    # Group by state
    grouped = group_pharmacies_by_state(unique_pharmacies)
    
    # Identify under-filled states
    underfilled = identify_underfilled_states(grouped, min_required=target_per_state)
    
    # Process each under-filled state
    for state, needed in underfilled.items():
        logger.info(f"Processing under-filled state: {state} (needs {needed} more pharmacies)")
        existing = grouped[state]
        updated = self_heal_state(state, existing, needed)
        grouped[state] = updated
    
    return grouped
