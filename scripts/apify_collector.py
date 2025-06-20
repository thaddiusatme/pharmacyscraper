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
import traceback

# Load environment variables from .env file
load_dotenv()

# Apify Actor ID for Google Maps Scraper
APIFY_ACTOR_ID = "nwua9Gu5YrADL7ZDj"


class ApifyCollector:
    """Collects pharmacy data using Apify's Google Maps Scraper."""
    
    def __init__(self, api_token: str = None, rate_limit_ms: int = 1000, output_dir: str = 'output'):
        """Initialize the Apify collector with an API token.
        
        Args:
            api_token: Apify API token. If not provided, will look for APIFY_TOKEN or APIFY_API_TOKEN environment variables.
            rate_limit_ms: Delay between API requests in milliseconds. Defaults to 1000ms.
            output_dir: Directory to save output files.
            
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
        self.rate_limit_ms = rate_limit_ms
        self.logger = logging.getLogger(__name__)
        self._last_request_time = 0
        self.output_dir = str(output_dir)  # Ensure it's a string
        self.output_dir_path = Path(self.output_dir)
        self.output_dir_path.mkdir(parents=True, exist_ok=True)
    
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
    
    def run_trial(self, config_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """Run a trial data collection using the provided configuration.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Dictionary mapping city names to lists of collected pharmacy records
            
        Raises:
            FileNotFoundError: If the config file is not found
            json.JSONDecodeError: If the config file contains invalid JSON
            ValueError: If the config is missing required fields
            Exception: For any other errors during execution
        """
        try:
            config = self._load_config(config_path)
            output_dir = config.get('output_dir', 'output')
            
            queries = self._generate_queries_from_config(config_path)
            
            state_dirs = list(set(q['state'] for q in queries))
            self._create_output_directories(output_dir, state_dirs)
            
            all_results = {}
            cities_processed = 0
            
            for query_info in queries:
                city_name = query_info['city']
                state = query_info['state']
                query = query_info['query']
                
                try:
                    for q in query.split('|'):
                        time_since_last_request = (time.time() * 1000) - self._last_request_time
                        if time_since_last_request < self.rate_limit_ms:
                            sleep_time = (self.rate_limit_ms - time_since_last_request) / 1000.0
                            time.sleep(sleep_time)
                        
                        results = self._process_city(city_name, state, q.strip(), output_dir, config)
                        if results:
                            all_results.setdefault(city_name, []).extend(results)
                        
                        self._last_request_time = time.time() * 1000
                        
                    cities_processed += 1
                    
                except Exception as e:
                    error_msg = f"Error processing {city_name}, {state}: {str(e)}"
                    self.logger.error(error_msg)
                    if "Actor not found" in str(e):
                        raise Exception(f"Error in run_trial: Actor not found") from e
                    continue
        
            return all_results
            
        except Exception as e:
            if "Actor not found" in str(e):
                raise Exception(f"Error in run_trial: Actor not found") from e
            raise Exception(f"Error in run_trial: {str(e)}") from e
    
    def _process_city(self, city: str, state: str, query: str, output_dir: str, config: dict) -> List[Dict[str, Any]]:
        """Process a single city's data collection.
        
        Args:
            city: Name of the city to process
            state: Name of the state where the city is located
            query: Search query to use
            output_dir: Directory to save results
            config: Configuration dictionary
            
        Returns:
            List of collected pharmacy records
        """
        try:
            state_name = state.title()
            
            run_input = {
                "searchStringsArray": [query],
                "maxCrawledPlacesPerSearch": config.get('max_results_per_query', 5),
                "language": "en",
                "maxRequestRetries": 2,
                "searchMatching": "all"
            }
            
            self.logger.info(f"Starting Apify actor for query: {query} in {city}, {state_name}")
            
            run = self.client.actor(APIFY_ACTOR_ID).call(run_input=run_input)
            
            dataset_id = run.get("defaultDatasetId")
            if not dataset_id:
                self.logger.error(f"No dataset ID found in run: {run}")
                return []
                
            dataset = self.client.dataset(dataset_id)
            
            try:
                items = list(dataset.iterate_items())
                
                if not items:
                    self.logger.warning(f"No results found for {query} in {city}, {state_name}")
                    return []
                    
                self.logger.info(f"Retrieved {len(items)} results for {query} in {city}, {state_name}")
                
                for item in items:
                    item['query'] = query
                    item['city'] = city
                    item['state'] = state_name
                    item['scraped_at'] = datetime.utcnow().isoformat()
                
                city_safe = city.lower().replace(' ', '_')
                state_safe = state_name.lower().replace(' ', '_')
                timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
                output_path = Path(output_dir) / f"{state_safe}_{city_safe}_{timestamp}.json"
                
                with open(output_path, 'w') as f:
                    json.dump(items, f, indent=2, default=str)
                
                self.logger.info(f"Saved {len(items)} results to {output_path}")
                return items
                
            except Exception as e:
                self.logger.error(f"Error processing results for {query} in {city}, {state_name}: {str(e)}")
                self.logger.debug(traceback.format_exc())
                return []
                
        except Exception as e:
            self.logger.error(f"Error processing {query} in {city}, {state_name}: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return []
    
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
            for state, cities in states_cities.items():
                for city in cities:
                    queries.append({
                        'query': f"independent pharmacy in {city}, {state}",
                        'state': state,
                        'city': city
                    })
        else:
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
                run_input = {
                    "searchStringsArray": [query],
                    "maxCrawledPlacesPerSearch": 30,  
                    "language": "en",
                    "maxRequestRetries": 3,
                    "searchMatching": "all"
                }
                
                run = self.client.actor(APIFY_ACTOR_ID).call(run_input=run_input)
                
                run_id = run["id"]
                
                run_client = self.client.run(run_id)
                run = run_client.wait_for_finish()
                
                dataset_id = run.get("defaultDatasetId")
                if not dataset_id:
                    self.logger.error(f"No dataset ID found in run: {run}")
                    continue
                    
                dataset = self.client.dataset(dataset_id)
                try:
                    results = []
                    pagination = dataset.list_items(limit=1000)
                    results = list(pagination)
                    
                    if not results:
                        self.logger.warning(f"No results found in dataset {dataset_id}")
                        continue
                        
                    self.logger.info(f"Retrieved {len(results)} results from dataset {dataset_id}")
                    
                except Exception as e:
                    self.logger.error(f"Error retrieving items from dataset {dataset_id}: {str(e)}")
                    self.logger.debug(traceback.format_exc())
                    continue
                
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
                self.logger.debug(traceback.format_exc())
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


def run_trial(config_path: Union[str, Path]) -> Dict[str, List[Dict[str, Any]]]:
    """Run a trial collection using the Apify Google Maps Scraper.
    
    This is a convenience function that creates an ApifyCollector instance
    and calls its run_trial method.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary mapping city names to lists of collected pharmacy items
        
    Raises:
        Exception: If there's an error during collection
    """
    collector = ApifyCollector()
    return collector.run_trial(config_path)


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
        states = ["California", "Texas"]
        queries = collector.generate_search_queries(states)
        results = collector.collect_pharmacies(queries)
        collector.save_results(results, "pharmacies.csv")
        print(f"Collected {len(results)} results")

if __name__ == "__main__":
    main()