"""
Pipeline orchestrator for the pharmacy scraper system.

This module contains the PipelineOrchestrator class that coordinates the entire
pharmacy data collection and processing pipeline.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
import pandas as pd

from ..api.apify_collector import ApifyCollector
from ..dedup_self_heal.dedup import remove_duplicates
from ..classification.classifier import Classifier
from ..verification.google_places import verify_pharmacy
from ..utils.api_usage_tracker import credit_tracker, APICreditTracker, CreditLimitExceededError
from ..classification.cache import load_from_cache, save_to_cache

logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """Configuration for the pipeline orchestrator."""
    # Data collection
    api_keys: Dict[str, str]
    max_results_per_query: int = 25
    
    # Paths
    output_dir: str = "output"
    cache_dir: str = "cache"
    
    # Classification
    classification_cache_dir: Optional[str] = None
    classification_threshold: float = 0.6
    
    # Verification
    verify_places: bool = True
    verification_confidence_threshold: float = 0.7
    
    # Budget
    max_budget: float = 100.0
    api_cost_limits: Dict[str, float] = field(
        default_factory=lambda: {
            "apify": 0.5,     # $0.50 per 1000 records
            "google_places": 0.02,  # $0.02 per lookup
            "perplexity": 0.01,     # $0.01 per classification
        }
    )
    
    # Data collection locations and queries
    locations: List[Dict[str, Union[str, List[str]]]] = field(
        default_factory=lambda: [
            {
                "state": "CA",
                "cities": ["San Francisco", "Los Angeles"],
                "queries": ["independent pharmacy"]
            }
        ]
    )

class PipelineState:
    """Tracks the state of the pipeline execution."""
    def __init__(self):
        self.current_phase = "initialized"
        self.phases_completed = set()
        self.metrics = {
            "pharmacies_collected": 0,
            "pharmacies_classified": 0,
            "pharmacies_verified": 0,
            "api_calls": {},
            "errors": []
        }
    
    def update_phase(self, phase_name: str):
        """
        Update the current phase and mark previous as completed.
        
        Args:
            phase_name: Name of the new phase
        """
        if self.current_phase and self.current_phase != "initialized":
            self.phases_completed.add(self.current_phase)
        self.current_phase = phase_name
        logger.info(f"Entering phase: {phase_name}")
    
    def complete_current_phase(self):
        """
        Mark the current phase as completed.
        This should be called when the pipeline completes to ensure the final phase is marked as completed.
        """
        if self.current_phase and self.current_phase != "initialized":
            self.phases_completed.add(self.current_phase)
            self.current_phase = "completed"

class PipelineOrchestrator:
    """
    Coordinates the entire pharmacy data collection and processing pipeline.
    
    The orchestrator manages the following phases:
    1. Data Collection (Apify)
    2. Deduplication
    3. Classification (Perplexity)
    4. Verification (Google Places)
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the pipeline orchestrator.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.state = PipelineState()
        self._setup_components()
    
    def _load_config(self, config_path: str) -> PipelineConfig:
        """Load and validate the pipeline configuration."""
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Set default values
        config_data.setdefault('output_dir', 'output')
        config_data.setdefault('cache_dir', 'cache')
        config_data.setdefault('verify_places', True)
        
        return PipelineConfig(**config_data)
    
    def _setup_components(self):
        """Initialize all pipeline components."""
        # Initialize API clients
        self.collector = ApifyCollector(
            api_key=self.config.api_keys.get('apify'),
            output_dir=self.config.output_dir,
        )
        
        # Initialize classifier
        self.classifier = Classifier()
        
        # Setup API budget tracking
        for service, cost in self.config.api_cost_limits.items():
            credit_tracker.set_cost_limit(service, cost)
        credit_tracker.budget = self.config.max_budget
    
    def run(self) -> Optional[Path]:
        """
        Execute the entire pipeline.
        
        Returns:
            Optional[Path]: The path to the output file on success, None on failure.
        """
        try:
            # Phase 1: Data Collection
            self.state.update_phase("data_collection")
            raw_pharmacies = self._collect_pharmacies()
            
            # Phase 2: Deduplication
            self.state.update_phase("deduplication")
            unique_pharmacies = self._deduplicate_pharmacies(raw_pharmacies)
            
            # Phase 3: Classification
            self.state.update_phase("classification")
            classified_pharmacies = self._classify_pharmacies(unique_pharmacies)
            
            # Phase 4: Verification
            if self.config.verify_places:
                self.state.update_phase("verification")
                verified_pharmacies = self._verify_pharmacies(classified_pharmacies)
            else:
                verified_pharmacies = classified_pharmacies
            
            # Save final results
            output_file = self._save_results(verified_pharmacies)
            
            # Mark the current phase as completed
            self.state.complete_current_phase()
            
            logger.info("Pipeline completed successfully!")
            return output_file
            
        except CreditLimitExceededError as e:
            logger.error(f"Budget exceeded: {e}")
            return None
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            return None
    
    def _collect_pharmacies(self) -> List[Dict]:
        """
        Collect pharmacies from configured locations and queries.
        
        Returns:
            List[Dict]: List of collected pharmacy records
            
        Raises:
            RuntimeError: If no locations are configured or collection fails
        """
        if not hasattr(self.config, 'locations') or not self.config.locations:
            raise RuntimeError("No locations configured for data collection")
            
        all_pharmacies = []
        
        for location_config in self.config.locations:
            state = location_config['state']
            cities = location_config.get('cities', [])
            queries = location_config.get('queries', ["independent pharmacy"])
            
            if not cities:
                logger.warning(f"No cities specified for state {state}, using state-level query")
                cities = [None]  # Will query at state level
            
            for city in cities:
                for query in queries:
                    try:
                        # Format location string
                        location = f"{city}, {state}" if city else state
                        logger.info(f"Collecting pharmacies for query: '{query}' in {location}")
                        
                        # Execute the query
                        results = self._execute_pharmacy_query(query, location)
                        
                        # Add location metadata
                        for result in results:
                            result.update({
                                'query': query,
                                'state': state,
                                'city': city
                            })
                        
                        all_pharmacies.extend(results)
                        logger.info(f"Collected {len(results)} pharmacies from {location}")
                        
                    except CreditLimitExceededError:
                        raise
                    except Exception as e:
                        logger.error(f"Failed to collect pharmacies for '{query}' in {location}: {e}")
                        continue
        
        if not all_pharmacies:
            raise RuntimeError("No pharmacies were collected. Check logs for details.")
            
        logger.info(f"Collected a total of {len(all_pharmacies)} pharmacies")
        return all_pharmacies
        
    def _execute_pharmacy_query(self, query: str, location: str) -> List[Dict]:
        """
        Execute a single pharmacy query and return results.
        
        Args:
            query: Search query string
            location: Location string (e.g., "San Francisco, CA")
            
        Returns:
            List of pharmacy records
            
        Raises:
            Exception: If query execution fails
        """
        cache_key = f"{query}_{location}".lower().replace(" ", "_")
        
        # Try to load from cache first
        cached_results = None
        try:
            cached_results = load_from_cache(cache_key, self.config.cache_dir)
        except Exception as e:
            logger.warning(f"Failed to load cache for query '{{query}}' in {{location}}. Error: {{e}}")

        if cached_results:
            logger.debug(f"Loaded {len(cached_results)} pharmacies from cache for '{query}' in {location}")
            return cached_results
        
        # Execute the query if not in cache
        try:
            # Track API usage
            with credit_tracker.track_usage('apify'):
                results = self.collector.run_trial(query, location)
                
                # Ensure results are in the expected format
                if not isinstance(results, list):
                    results = [results] if results else []
                    
                # Cache the results
                save_to_cache(results, cache_key, self.config.cache_dir)
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to execute query '{query}' in {location}: {e}")
            raise
    
    def _deduplicate_pharmacies(self, pharmacies: List[Dict]) -> List[Dict]:
        """Remove duplicate pharmacy records."""
        df = pd.DataFrame(pharmacies)
        df_deduped = remove_duplicates(df)
        return df_deduped.to_dict('records')
    
    def _classify_pharmacies(self, pharmacies: List[Dict]) -> List[Dict]:
        """Classify pharmacies using the configured classifier."""
        results = []
        for pharmacy in pharmacies:
            try:
                classification_result = self.classifier.classify_pharmacy(pharmacy)
                pharmacy["classification"] = classification_result.to_dict()
                results.append(pharmacy)
            except Exception as e:
                logger.warning(f"Failed to classify pharmacy {pharmacy.get('id')}: {e}")
                results.append({**pharmacy, "classification_error": str(e)})
        return results
    
    def _verify_pharmacies(self, pharmacies: List[Dict]) -> List[Dict]:
        """Verify pharmacy information using Google Places API."""
        results = []
        for pharmacy in pharmacies:
            try:
                verification = verify_pharmacy(pharmacy)
                results.append({**pharmacy, "verification": verification})
            except Exception as e:
                logger.warning(f"Failed to verify pharmacy {pharmacy.get('id')}: {e}")
                results.append({**pharmacy, "verification_error": str(e)})
        return results
    
    def _save_results(self, pharmacies: List[Dict]) -> Path:
        """
        Save the final results to disk as JSON and CSV.
        
        Args:
            pharmacies: List of pharmacy records to save.
            
        Returns:
            Path to the saved JSON file.
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define output file paths
        json_output_file = output_dir / 'pharmacies.json'
        csv_output_file = output_dir / 'pharmacies.csv'

        # Save as JSON
        with open(json_output_file, 'w') as f:
            json.dump(pharmacies, f, indent=2)
        
        # Save as CSV
        df = pd.DataFrame(pharmacies)
        df.to_csv(csv_output_file, index=False)
        
        logger.info(f"Saved {len(pharmacies)} pharmacies to {output_dir}")
        return json_output_file