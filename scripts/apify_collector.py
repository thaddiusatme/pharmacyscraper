import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
from datetime import datetime
import pandas as pd
from apify_client import ApifyClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Apify Actor ID for Google Maps Scraper
APIFY_ACTOR_ID = "nwua9Gu5YrADL7ZDj"


class ApifyCollector:
    """Collects pharmacy data using Apify's Google Maps Scraper."""
    
    def __init__(self, api_token: str = None):
        """Initialize the Apify collector with an API token.
        
        Args:
            api_token: Apify API token. If not provided, will look for APIFY_TOKEN or APIFY_API_TOKEN environment variables.
            
        Raises:
            ValueError: If no API token is provided and not found in environment variables.
        """
        self.api_token = api_token or os.getenv('APIFY_TOKEN') or os.getenv('APIFY_API_TOKEN')
        if not self.api_token:
            raise ValueError(
                "Apify API token is required. "
                "Either pass it to the constructor or set APIFY_TOKEN/APIFY_API_TOKEN environment variable."
            )
        
        self.client = ApifyClient(self.api_token)
        self.logger = logging.getLogger(__name__)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load trial configuration from a JSON file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Dictionary containing the configuration
        """
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                return config_data.get('trial_run_20240619', {})
        except (json.JSONDecodeError, FileNotFoundError) as e:
            self.logger.error(f"Error loading config file: {e}")
            raise

    def _generate_queries_from_config(self, config_path: str) -> List[Dict[str, str]]:
        """Generate search queries from the trial configuration.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            List of query dictionaries with 'query', 'state', and 'city' keys
        """
        config = self._load_config(config_path)
        queries = []
        
        for state, state_data in config.get('states', {}).items():
            for city_data in state_data.get('cities', []):
                city_name = city_data.get('name', '')
                for query in city_data.get('queries', []):
                    queries.append({
                        'query': query,
                        'state': state,
                        'city': city_name
                    })
        
        return queries
    
    def _create_output_directories(self, output_dir: str, state_dirs: List[str]) -> None:
        """Create output directories for the trial run.
        
        Args:
            output_dir: Base output directory
            state_dirs: List of state directories to create
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for state_dir in state_dirs:
            (output_path / state_dir.lower().replace(' ', '_')).mkdir(exist_ok=True)
    
    def run_trial(self, config_path: str) -> List[Dict[str, Any]]:
        """Run a trial data collection using the provided configuration.
        
        Args:
            config_path: Path to the trial configuration file
            
        Returns:
            List of collected pharmacy records
            
        Raises:
            ValueError: If the configuration is invalid or empty
            Exception: If there's an error with the Apify actor
        """
        # Load configuration
        config = self._load_config(config_path)
        if not config:
            raise ValueError("Invalid or empty configuration")
        
        # Set up output directories
        output_dir = config.get('output_dir', 'data/raw/trial_run')
        state_dirs = list(config.get('states', {}).keys())
        self._create_output_directories(output_dir, state_dirs)
        
        # Generate queries
        queries = self._generate_queries_from_config(config_path)
        if not queries:
            self.logger.warning("No queries generated from configuration")
            return []
        
        # Set rate limiting
        rate_limit_ms = config.get('rate_limit_ms', 50)
        max_results = config.get('max_results_per_query', 30)
        
        all_results = []
        
        for query_info in queries:
            query = query_info['query']
            state = query_info['state']
            city = query_info['city']
            
            self.logger.info(f"Searching for: {query} in {city}, {state}")
            
            try:
                # Prepare the Actor input
                run_input = {
                    "searchStringsArray": [query],
                    "maxCrawledPlacesPerSearch": max_results,
                    "language": "en",
                    "maxRequestRetries": 3,
                    "searchMatching": "all"
                }
                
                # Run the Actor - using the correct actor ID for Google Maps Scraper
                run = self.client.actor(APIFY_ACTOR_ID).start(run_input=run_input)
                run.wait_for_finish()
                
                # Get results
                dataset = self.client.dataset(run["defaultDatasetId"])
                results = list(dataset.iterate_items())
                
                # Add metadata to results
                for result in results:
                    result.update({
                        'search_query': query,
                        'search_state': state,
                        'search_city': city,
                        'collected_at': datetime.utcnow().isoformat()
                    })
                
                all_results.extend(results)
                
                # Save results to state/city file
                if results:
                    state_dir = state.lower().replace(' ', '_')
                    city_file = city.lower().replace(' ', '_') + '.json'
                    output_path = Path(output_dir) / state_dir / city_file
                    
                    with open(output_path, 'w') as f:
                        json.dump(results, f, indent=2)
                
                # Rate limiting
                time.sleep(rate_limit_ms / 1000.0)
                
            except Exception as e:
                self.logger.error(f"Error processing query '{query}': {e}")
                # Re-raise the exception to fail fast in case of critical errors
                if "Actor not found" in str(e):
                    raise Exception(f"Apify actor error: {e}")
                continue
        
        return all_results
    
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
        
    def collect_pharmacies(self, queries: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Collect pharmacy data from Apify for the given queries.
        
        Args:
            queries: List of search queries with state and city information
            
        Returns:
            List of pharmacy records with metadata
        """
        all_results = []
        
        for query_info in queries:
            query = query_info['query']
            state = query_info['state']
            city = query_info['city']
            
            self.logger.info(f"Searching for: {query} in {city}, {state}")
            
            try:
                # Prepare the Actor input
                run_input = {
                    "searchStringsArray": [query],
                    "maxCrawledPlacesPerSearch": 30,  # Default max results
                    "language": "en",
                    "maxRequestRetries": 3,
                    "searchMatching": "all"
                }
                
                # Run the Actor
                run = self.client.actor(APIFY_ACTOR_ID).start(run_input=run_input)
                run.wait_for_finish()
                
                # Get results
                dataset = self.client.dataset(run["defaultDatasetId"])
                results = list(dataset.iterate_items())
                
                # Add metadata to results
                for result in results:
                    result.update({
                        'search_query': query,
                        'search_state': state,
                        'search_city': city,
                        'collected_at': datetime.utcnow().isoformat()
                    })
                
                all_results.extend(results)
                
            except Exception as e:
                self.logger.error(f"Error processing query '{query}': {e}")
                if "Actor not found" in str(e):
                    raise Exception(f"Apify actor error: {e}")
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
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Apify data collection')
    parser.add_argument('--trial-config', help='Path to trial configuration file')
    args = parser.parse_args()
    
    collector = ApifyCollector()
    
    if args.trial_config:
        print(f"Running trial with config: {args.trial_config}")
        results = collector.run_trial(args.trial_config)
        print(f"Collected {len(results)} results")
    else:
        # Default behavior (existing code)
        states = ["California", "Texas"]
        queries = collector.generate_search_queries(states)
        results = collector.collect_pharmacies(queries)
        collector.save_results(results, "pharmacies.csv")
        print(f"Collected {len(results)} results")

if __name__ == "__main__":
    main()