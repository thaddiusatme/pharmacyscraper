import os
from typing import List, Dict, Any, Optional, Union
import logging
from datetime import datetime
import pandas as pd
from apify_client import ApifyClient

class ApifyCollector:
    """Collects pharmacy data using Apify's Google Maps Scraper."""
    
    def __init__(self, api_token: str = None):
        """Initialize the Apify collector with an API token.
        
        Args:
            api_token: Apify API token. If not provided, will look for APIFY_TOKEN environment variable.
        """
        self.api_token = api_token or os.getenv('APIFY_TOKEN')
        if not self.api_token:
            raise ValueError(
                "Apify API token is required. "
                "Either pass it to the constructor or set APIFY_TOKEN environment variable."
            )
        
        self.client = ApifyClient(self.api_token)
        self.logger = logging.getLogger(__name__)
        
    def generate_search_queries(self, states_cities: Union[Dict[str, List[str]], List[str]], 
                             cities_per_state: int = None) -> List[Dict[str, str]]:
        """Generate search queries for each state and city combination.
        
        Args:
            states_cities: Either a dict mapping states to lists of cities, or a list of state names
            cities_per_state: Number of cities to use per state (only used if states_cities is a list)
            
        Returns:
            List of search query dictionaries with 'query', 'state', and 'city' keys
        """
        queries = []
        
        if isinstance(states_cities, dict):
            # Handle dictionary input {state: [cities]}
            for state, cities in states_cities.items():
                for city in cities:
                    queries.append({
                        'query': f"independent pharmacy in {city}, {state}",
                        'state': state,
                        'city': city
                    })
        else:
            # Handle list of states
            states = states_cities
            for state in states:
                queries.append({
                    'query': f"independent pharmacy in {state}",
                    'state': state,
                    'city': ''
                })
                
        return queries
        
    def collect_pharmacies(self, queries: List[Dict[str, str]], max_results: int = 100) -> List[Dict[str, Any]]:
        """Collect pharmacy data from Apify's Google Maps Scraper.
        
        Args:
            queries: List of search query dictionaries
            max_results: Maximum number of results to return per query
            
        Returns:
            List of pharmacy records
        """
        all_results = []
        
        for query_info in queries:
            query = query_info['query']
            self.logger.info(f"Searching for: {query}")
            
            # Prepare the Actor input
            run_input = {
                "queries": query,
                "maxPlacesPerQuery": max_results,
                "language": "en",
                "maxRequestRetries": 3,
            }
            
            try:
                # Run the Actor and wait for it to finish
                run = self.client.actor("apify/google-maps-scraper").start(run_input=run_input)
                run.wait_for_finish()
                
                # Get the results from the run's dataset
                dataset = self.client.dataset(run["defaultDatasetId"])
                results = list(dataset.iterate_items())
                
                # Add state and city info to each result
                for result in results:
                    result.update({
                        'search_state': query_info.get('state', ''),
                        'search_city': query_info.get('city', '')
                    })
                
                all_results.extend(results)
                
            except Exception as e:
                self.logger.error(f"Error searching for '{query}': {str(e)}")
                continue
                
        return all_results
        
    def _is_valid_pharmacy(self, pharmacy: Dict[str, Any], filter_chains: bool = True) -> bool:
        """Check if a pharmacy is valid (not a chain).
        
        Args:
            pharmacy: Pharmacy data dictionary
            filter_chains: If True, filter out chain pharmacies
            
        Returns:
            bool: True if pharmacy is valid, False otherwise
        """
        if not pharmacy or 'name' not in pharmacy:
            return False
            
        if not filter_chains:
            return True
            
        name = pharmacy['name'].lower()
        chain_keywords = [
            'cvs', 'walgreens', 'rite aid', 'walmart', 'wal-mart',
            'cvs pharmacy', 'walgreens pharmacy', 'rite aid pharmacy',
            'duane reade', 'riteaid', 'walgreens.com', 'cvs.com',
            'riteaid.com', 'wal-mart pharmacy', 'walmart pharmacy'
        ]
        
        return not any(keyword in name for keyword in chain_keywords)
    
    def filter_chain_pharmacies(self, pharmacies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out chain pharmacies from the results.
        
        Args:
            pharmacies: List of pharmacy records
            
        Returns:
            Filtered list of independent pharmacies
        """
        return [p for p in pharmacies if self._is_valid_pharmacy(p)]
    
    def save_results(self, pharmacies: List[Dict[str, Any]], output_path: str) -> None:
        """Save pharmacy data to a CSV file.
        
        Args:
            pharmacies: List of pharmacy records
            output_path: Path to save the CSV file
        """
        if not pharmacies:
            self.logger.warning("No pharmacy data to save")
            return
            
        df = pd.DataFrame(pharmacies)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        self.logger.info(f"Saved {len(pharmacies)} pharmacies to {output_path}")


def main():
    """Example usage of the ApifyCollector class."""
    import os
    from dotenv import load_dotenv
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Initialize the collector
    collector = ApifyCollector()
    
    # Example states to search
    states = ["California", "New York", "Texas"]
    
    # Generate search queries
    queries = collector.generate_search_queries(states)
    
    # Collect pharmacy data
    pharmacies = collector.collect_pharmacies(queries)
    
    # Filter out chain pharmacies
    independent_pharmacies = collector.filter_chain_pharmacies(pharmacies)
    
    # Save results
    output_dir = os.path.join('data', 'raw')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'apify_pharmacies.csv')
    collector.save_results(independent_pharmacies, output_path)
    print(f"Saved {len(independent_pharmacies)} independent pharmacies to {output_path}")

if __name__ == "__main__":
    main()