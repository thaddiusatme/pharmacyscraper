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
    
    def _load_config(self, config_path: str) -> dict:
        """Load and validate the configuration file.
        
        Args:
            config_path: Path to the JSON configuration file
            
        Returns:
            dict: The loaded and validated configuration
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # The config is nested under a trial_run_* key
            trial_keys = [k for k in config.keys() if k.startswith('trial_run_')]
            if not trial_keys:
                raise ValueError("No trial configuration found in the config file")
                
            # Use the first trial config found
            trial_config = config[trial_keys[0]]
            
            # Set defaults for required fields
            trial_config.setdefault('output_dir', 'data/raw/trial')
            trial_config.setdefault('max_cities_per_run', 2)
            trial_config.setdefault('max_results_per_query', 5)
            trial_config.setdefault('rate_limit_ms', 2000)
            
            return trial_config
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in config file: {e}")
            raise
        except FileNotFoundError:
            self.logger.error(f"Config file not found: {config_path}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
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
    
    def run_trial(self, config_path: str) -> None:
        """Run a trial data collection using the provided configuration."""
        try:
            config = self._load_config(config_path)
            self.logger.info(f"Loaded configuration: {json.dumps(config, indent=2, default=str)}")
            
            output_dir = config.get('output_dir', 'data/raw/trial')
            max_cities = config.get('max_cities_per_run', 2)
            
            # Create output directory if it doesn't exist
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Process cities in batches to control costs
            cities_processed = 0
            
            # Extract states and cities from the nested structure
            states_config = config.get('states', {})
            for state, state_data in states_config.items():
                if cities_processed >= max_cities:
                    self.logger.info(f"Reached maximum of {max_cities} cities for this run")
                    break
                    
                cities = state_data.get('cities', [])
                for city_data in cities:
                    if cities_processed >= max_cities:
                        break
                        
                    city_name = city_data.get('name', '').replace('_', ' ').title()
                    queries = city_data.get('queries', [])
                    
                    if not queries:
                        self.logger.warning(f"No queries found for city: {city_name}")
                        continue
                        
                    self.logger.info(f"Processing city: {city_name}, {state.title()}")
                    self._process_city(city_name, state.title(), queries[0], output_dir, config)
                    cities_processed += 1
                    
                    # Add delay between cities to avoid rate limiting
                    if cities_processed < max_cities:  # No need to sleep after the last city
                        time.sleep(5)  # 5 second delay between cities
                        
        except Exception as e:
            self.logger.error(f"Error in run_trial: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise
    
    def _process_city(self, city: str, state: str, query: str, output_dir: str, config: dict) -> None:
        """Process a single city's data collection."""
        try:
            # Prepare the Actor input with configuration
            run_input = {
                "searchStringsArray": [query],
                "maxCrawledPlacesPerSearch": config.get('max_results_per_query', 5),
                "language": "en",
                "maxRequestRetries": 2,
                "searchMatching": "all"
            }
            
            self.logger.info(f"Starting Apify actor for query: {query}")
            
            # Run the Actor
            run = self.client.actor(APIFY_ACTOR_ID).call(run_input=run_input)
            run_id = run["id"]
            
            # Wait for the run to finish
            run_client = self.client.run(run_id)
            run = run_client.wait_for_finish()
            
            # Get results
            dataset_id = run.get("defaultDatasetId")
            if not dataset_id:
                self.logger.error(f"No dataset ID found in run: {run}")
                return
                
            dataset = self.client.dataset(dataset_id)
            
            # Get items from the dataset
            try:
                # Get items from the dataset
                items = []
                
                # Get the first page of results
                page = dataset.list_items(limit=config.get('max_results_per_query', 5))
                
                # Access the items using the 'items' attribute
                items = page.items if hasattr(page, 'items') else []
                
                if not items:
                    self.logger.warning(f"No results found for {query}")
                    return
                    
                self.logger.info(f"Retrieved {len(items)} results for {query}")
                
                # Save results
                city_file = city.lower().replace(' ', '_') + '.json'
                output_path = Path(output_dir) / state.lower() / city_file
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'w') as f:
                    json.dump(items, f, indent=2, default=str)
                
                self.logger.info(f"Saved results to {output_path}")
                
            except Exception as e:
                self.logger.error(f"Error processing results for {query}: {str(e)}")
                import traceback
                self.logger.debug(traceback.format_exc())
                
        except Exception as e:
            self.logger.error(f"Error processing {query}: {str(e)}")
            import traceback
            self.logger.debug(traceback.format_exc())
    
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
                try:
                    run = self.client.actor(APIFY_ACTOR_ID).call(run_input=run_input)
                    
                    # Get the run ID from the response
                    run_id = run["id"]
                    
                    # Wait for the run to finish using the run client
                    run_client = self.client.run(run_id)
                    run = run_client.wait_for_finish()
                    
                    # Get results
                    dataset_id = run.get("defaultDatasetId")
                    if not dataset_id:
                        self.logger.error(f"No dataset ID found in run: {run}")
                        continue
                        
                    dataset = self.client.dataset(dataset_id)
                    try:
                        # Get items with pagination using the correct parameters
                        results = []
                        pagination = dataset.list_items(limit=1000)
                        results = list(pagination)
                        
                        if not results:
                            self.logger.warning(f"No results found in dataset {dataset_id}")
                            continue
                            
                        self.logger.info(f"Retrieved {len(results)} results from dataset {dataset_id}")
                        
                    except Exception as e:
                        self.logger.error(f"Error retrieving items from dataset {dataset_id}: {str(e)}")
                        import traceback
                        self.logger.debug(traceback.format_exc())
                        continue
                    
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
                    self.logger.error(f"Error processing query '{query}': {str(e)}")
                    import traceback
                    self.logger.debug(traceback.format_exc())
                    continue
            
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
        collector.run_trial(args.trial_config)
    else:
        # Default behavior (existing code)
        states = ["California", "Texas"]
        queries = collector.generate_search_queries(states)
        results = collector.collect_pharmacies(queries)
        collector.save_results(results, "pharmacies.csv")
        print(f"Collected {len(results)} results")

if __name__ == "__main__":
    main()