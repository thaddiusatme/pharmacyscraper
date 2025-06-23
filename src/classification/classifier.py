"""
This module defines the Classifier for identifying independent pharmacies.
"""
import pandas as pd
from typing import Dict, Optional
import logging

from .perplexity_client import PerplexityClient

logger = logging.getLogger(__name__)

class Classifier:
    """
    Classifies pharmacies using a combination of rules and a Perplexity API client.
    """

    def __init__(self, perplexity_client_or_api_key=None, cache_dir: Optional[str] = "data/cache/classification", force_reclassification: bool = False):
        """
        Initializes the Classifier.

        Args:
            perplexity_client_or_api_key: Either a PerplexityClient instance or an API key string.
            cache_dir: The directory to store cache files for the Perplexity client (only used if creating new client).
            force_reclassification: If True, ignores cached results and re-queries the API.
        """
        if isinstance(perplexity_client_or_api_key, PerplexityClient):
            # Use the provided PerplexityClient instance
            self.perplexity_client = perplexity_client_or_api_key
            self.perplexity_client.force_reclassification = force_reclassification
        else:
            # Create a new PerplexityClient with the provided api_key (or None)
            self.perplexity_client = PerplexityClient(
                api_key=perplexity_client_or_api_key, 
                cache_dir=cache_dir, 
                force_reclassification=force_reclassification
            )
        # Simple rule-based check for major chains
        self.chain_keywords = ['cvs', 'walgreens', 'rite aid', 'walmart', 'costco', 'kroger', 'safeway', 'albertsons']

    def _is_chain_by_name(self, name: str) -> bool:
        """
        A simple rule-based check to quickly identify major chains by name.
        """
        if not isinstance(name, str):
            return False
        name_lower = name.lower()
        return any(keyword in name_lower for keyword in self.chain_keywords)

    def classify_pharmacies(self, pharmacies_df: pd.DataFrame) -> pd.DataFrame:
        """
        Classifies a DataFrame of pharmacies, adding classification columns.

        This method applies both rule-based and LLM-based classification.

        Args:
            pharmacies_df: DataFrame with pharmacy data.

        Returns:
            DataFrame with added classification columns.
        """
        results = []
        for _, row in pharmacies_df.iterrows():
            place_id = row.get('placeId')
            name = row.get('title', '')

            # Rule-based pre-classification for obvious chains
            # Temporarily disabled to test LLM caching
            if False: # self._is_chain_by_name(name):
                results.append({
                    'placeId': place_id,
                    'classification': 'chain',
                    'is_independent': False,
                    'is_compounding': False, # Assume major chains are not primarily compounding
                    'confidence': 1.0,
                    'source': 'rule-based',
                    'method': 'keyword_match'
                })
                continue
            
            # LLM-based classification for all other cases
            pharmacy_data = row.to_dict()
            api_result = self.perplexity_client.classify_pharmacy(pharmacy_data)

            if api_result:
                classification = api_result.get('classification', 'unknown')
                is_independent = classification == 'independent'
                
                results.append({
                    'placeId': place_id,
                    'classification': classification,
                    'is_independent': is_independent,
                    'is_compounding': api_result.get('is_compounding', False),
                    'confidence': api_result.get('confidence', 0.0),
                    'source': 'perplexity',
                    'method': 'llm'
                })
            else:
                # Handle cases where the API call fails or returns an invalid result
                results.append({
                    'placeId': place_id,
                    'classification': 'error',
                    'is_independent': None,
                    'is_compounding': None,
                    'confidence': 0.0,
                    'source': 'error',
                    'method': 'api_failure'
                })
        
        if not results:
            return pd.DataFrame(columns=['placeId', 'classification', 'is_independent', 'is_compounding', 'confidence', 'source', 'method'])

        return pd.DataFrame(results)

    def classify_pharmacy(self, pharmacy_data: Dict) -> Optional[Dict]:
        """
        Classifies a single pharmacy.

        Args:
            pharmacy_data: Dictionary with pharmacy data.

        Returns:
            Dictionary with classification results or None if classification fails.
        """
        place_id = pharmacy_data.get('placeId')
        name = pharmacy_data.get('title', '')

        # Rule-based pre-classification for obvious chains
        # Temporarily disabled to test LLM caching
        if False: # self._is_chain_by_name(name):
            return {
                'placeId': place_id,
                'classification': 'chain',
                'is_independent': False,
                'is_compounding': False,  # Assume major chains are not primarily compounding
                'confidence': 1.0,
                'source': 'rule-based',
                'method': 'keyword_match'
            }

        # LLM-based classification
        api_result = self.perplexity_client.classify_pharmacy(pharmacy_data)

        if api_result:
            classification = api_result.get('classification', 'unknown')
            is_independent = classification == 'independent'
            
            return {
                'placeId': place_id,
                'classification': classification,
                'is_independent': is_independent,
                'is_compounding': api_result.get('is_compounding', False),
                'confidence': api_result.get('confidence', 0.0),
                'source': 'perplexity',
                'method': 'llm'
            }
        else:
            # Handle cases where the API call fails or returns an invalid result
            return {
                'placeId': place_id,
                'classification': 'error',
                'is_independent': None,
                'is_compounding': None,
                'confidence': 0.0,
                'source': 'error',
                'method': 'api_failure'
            }