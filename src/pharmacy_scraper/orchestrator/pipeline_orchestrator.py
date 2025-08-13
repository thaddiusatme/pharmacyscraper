"""
Pipeline orchestrator for the pharmacy scraper system.

This module contains the PipelineOrchestrator class that coordinates the entire
pharmacy data collection and processing pipeline.
"""

import json
 
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
import pandas as pd

# Normalization utilities
from pharmacy_scraper.normalization.address import normalize_address
from pharmacy_scraper.normalization.phone import normalize_phone

# Contact enrichment
from pharmacy_scraper.enrichment.npi_contact import enrich_contact_from_npi

from ..api.apify_collector import ApifyCollector
from ..dedup_self_heal.dedup import remove_duplicates
from ..classification.classifier import Classifier
from ..classification.models import PharmacyData
from ..verification.google_places import verify_pharmacy
from ..utils.api_usage_tracker import credit_tracker, APICreditTracker, CreditLimitExceededError
from ..classification.cache import load_from_cache, save_to_cache
from .state_manager import StateManager
from pharmacy_scraper.pipeline.plugin_pipeline import run_pipeline
from pharmacy_scraper.config.loader import load_config as load_config_file
from pharmacy_scraper.observability.logging import get_structured_logger, bind_context
import uuid

logger = get_structured_logger(__name__)

# Schema v2 fields to ensure presence in outputs
# Base fields (Iteration 4 guaranteed tail order)
SCHEMA_V2_BASE_FIELDS = [
    # Address fields (US-first)
    "address_line1",
    "address_line2",
    "city",
    "state",
    "postal_code",
    "country_iso2",
    # Phone normalization
    "phone_e164",
    "phone_national",
    # Contact enrichment (Phase 1 minimal set for serialization)
    "contact_name",
    "contact_email",
    "contact_role",
]

# Remaining fields introduced subsequently
SCHEMA_V2_REMAINING_FIELDS = [
    "contact_source",
    "contact_email_source",
]

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

    # Crosswalk schema v2: business type to scrape (e.g., "pharmacy", "clinic")
    business_type: str = "pharmacy"
    # Crosswalk schema v2: optional search terms override
    search_terms: List[str] = field(default_factory=list)

    # Feature flags (default off); loader may coerce to 0/1
    EMAIL_DISCOVERY_ENABLED: int = 0
    INTERNATIONAL_ENABLED: int = 0

    # Plugin-driven pipeline (optional)
    plugin_mode: bool = False
    # Raw plugin config section passed to registry.build_from_config
    plugins: Optional[Dict[str, List[str]]] = None
    # Per-plugin config mapping by class name
    plugin_config: Optional[Dict[str, Dict[str, Any]]] = None

class PipelineOrchestrator:

    """
    Coordinates the entire pharmacy data collection and processing pipeline.
    
    The orchestrator manages the following phases:
    1. Data Collection (Apify)
    2. Deduplication
    3. Classification (Perplexity)
    4. Verification (Google Places)
    """
    
    STAGES = ["data_collection", "deduplication", "classification", "verification"]

    def __init__(self, config_path: str, db_path: str = "pipeline_state.db"):
        """
        Initialize the pipeline orchestrator.

        Args:
            config_path: Path to the configuration file.
            db_path: Path to the SQLite database for state management.
        """
        self.config = self._load_config(config_path)
        self.state_manager = StateManager(db_path=db_path)
        self._setup_components()
    
    def _load_config(self, config_path: str) -> PipelineConfig:
        """Load and validate the pipeline configuration.
        Uses the central config loader to apply env var substitution,
        defaults, and minimal schema validation.
        """
        cfg_dict = load_config_file(config_path)
        return PipelineConfig(**cfg_dict)
    
    def _setup_components(self):
        """Initialize all pipeline components."""
        # In plugin mode, adapters and pipeline handle components
        if not getattr(self.config, "plugin_mode", False):
            # Initialize API clients
            self.collector = ApifyCollector(
                api_key=self.config.api_keys.get('apify') if self.config.api_keys else None,
                output_dir=self.config.output_dir,
            )
            # Initialize classifier
            self.classifier = Classifier()
        
        # Setup API budget tracking
        for service, cost in self.config.api_cost_limits.items():
            credit_tracker.set_cost_limit(service, cost)
        credit_tracker.budget = self.config.max_budget
        # Set daily limit to 80% of the total budget to allow for multiple runs
        credit_tracker.daily_limit = self.config.max_budget * 0.8
    
    def _execute_stage(self, stage_name: str, stage_fn: Callable, *args, **kwargs) -> Any:
        """
        Executes a pipeline stage with state tracking and resume logic.
        - Checks if the stage is already completed.
        - Saves stage output upon completion.
        - Loads stage output if skipping.
        """
        output_dir = Path(self.config.output_dir)
        stage_output_file = output_dir / f"stage_{stage_name}_output.json"

        stage_logger = bind_context(getattr(self, "_run_logger", logger), {"stage": stage_name})

        if self.state_manager.get_task_status(stage_name) == 'completed':
            stage_logger.info("stage_skipped", extra={"event": "stage_skipped"})
            if not stage_output_file.exists():
                error_msg = f"Cannot skip stage '{stage_name}': Output file not found at {stage_output_file}"
                stage_logger.error(error_msg, extra={"event": "stage_error"})
                raise FileNotFoundError(error_msg)
            with open(stage_output_file, 'r') as f:
                return json.load(f)

        start = time.time()
        stage_logger.info("stage_start", extra={"event": "stage_start"})
        self.state_manager.update_task_status(stage_name, 'in_progress')

        try:
            result = stage_fn(*args, **kwargs)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Debug the result type
            stage_logger.debug(f"Stage '{stage_name}' returned type: {type(result)}")
            
            with open(stage_output_file, 'w') as f:
                if hasattr(result, 'to_dict') and callable(getattr(result, 'to_dict')):
                    # Handle pandas DataFrame
                    json.dump(result.to_dict('records'), f, indent=2)
                elif isinstance(result, list):
                    json.dump(result, f, indent=2)
                else:
                    json.dump(result, f, indent=2)

            self.state_manager.update_task_status(stage_name, 'completed')
            duration_ms = int((time.time() - start) * 1000)
            # Try to infer result count
            result_count = None
            try:
                if isinstance(result, list):
                    result_count = len(result)
                elif hasattr(result, "__len__"):
                    result_count = len(result)  # type: ignore[arg-type]
            except Exception:
                result_count = None
            stage_logger.info("stage_completed", extra={"event": "stage_completed", "duration_ms": duration_ms, "result_count": result_count})
            return result
        except Exception as e:
            self.state_manager.update_task_status(stage_name, 'failed')
            stage_logger.error(f"Stage '{stage_name}' failed: {e}", extra={"event": "stage_error"}, exc_info=True)
            raise

    def run(self) -> Optional[Path]:
        """
        Execute the entire pipeline with resume capability.

        Returns:
            Optional[Path]: The path to the output file on success, None on failure.
        """
        try:
            # Bind a run-scoped structured logger with run_id
            run_id = str(uuid.uuid4())
            self._run_logger = get_structured_logger(__name__, base_context={"run_id": run_id})
            self._run_logger.info("run_start", extra={"event": "run_start"})

            # If plugin mode is enabled, run the plugin-driven pipeline instead
            if getattr(self.config, "plugin_mode", False):
                # Construct a dict-shaped config expected by run_pipeline
                plugin_cfg = {
                    "plugins": self.config.plugins or {},
                    "plugin_config": self.config.plugin_config or {},
                }
                # Build a minimal query if available from locations, else empty
                query: Dict[str, Any] = {"business_type": getattr(self.config, "business_type", "pharmacy")}
                if self.config.locations:
                    # Use first location/query as a simple seed
                    loc0 = self.config.locations[0]
                    query = {**query, "location": loc0}
                results = run_pipeline(plugin_cfg, query=query)
                output_file = self._save_results(results)
                self._run_logger.info("run_completed", extra={"event": "run_completed"})
                return output_file

            raw_pharmacies = self._execute_stage("data_collection", self._collect_pharmacies)

            unique_pharmacies = self._execute_stage(
                "deduplication", self._deduplicate_pharmacies, pharmacies=raw_pharmacies
            )

            classified_pharmacies = self._execute_stage(
                "classification", self._classify_pharmacies, pharmacies=unique_pharmacies
            )

            if self.config.verify_places:
                verified_pharmacies = self._execute_stage(
                    "verification", self._verify_pharmacies, pharmacies=classified_pharmacies
                )
            else:
                logger.info("Skipping verification stage as per configuration.")
                verified_pharmacies = classified_pharmacies

            output_file = self._save_results(verified_pharmacies)
            self._run_logger.info("run_completed", extra={"event": "run_completed"})
            return output_file

        except CreditLimitExceededError as e:
            getattr(self, "_run_logger", logger).error(f"Budget exceeded: {e}", extra={"event": "run_error"})
            return None
        except Exception as e:
            getattr(self, "_run_logger", logger).error(f"Pipeline failed during execution: {e}", extra={"event": "run_error"}, exc_info=False)
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
            # Back-compat: allow location_config to be a simple string like "AK" or "Anchorage, AK"
            if isinstance(location_config, str):
                # If the string contains a comma, treat it as "City, State"; otherwise, assume it's a state
                parts = [p.strip() for p in location_config.split(',')]
                if len(parts) == 2:
                    state = parts[1]
                    cities = [parts[0]]
                else:
                    state = parts[0]
                    cities = []
                queries = ["independent pharmacy"]
            else:
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
        business_type = getattr(self.config, "business_type", "pharmacy")
        base_key = f"{query}_{location}".lower().replace(" ", "_")
        # Backward compatibility: do not change cache key for default business_type
        cache_key = (
            f"{business_type}:{base_key}" if business_type and business_type != "pharmacy" else base_key
        )
        
        # Try to load from cache first
        cached_results = None
        try:
            cached_results = load_from_cache(cache_key, self.config.cache_dir)
        except Exception as e:
            logger.warning(f"Failed to load cache for query '{{query}}' in {{location}}. Error: {{e}}")

        if cached_results:
            logger.debug(f"Loaded {len(cached_results)} pharmacies from cache for '{query}' in {location}")
            return cached_results
        
        # Backward lookup: if using a typed cache key and it's missing, try legacy untyped key
        if business_type and business_type != "pharmacy":
            try:
                legacy_results = load_from_cache(base_key, self.config.cache_dir)
            except Exception as e:
                logger.warning(
                    f"Failed legacy cache lookup for query '{query}' in {location}. Error: {e}"
                )
                legacy_results = None
            if legacy_results:
                logger.info(
                    "cache_backward_lookup_hit",
                    extra={
                        "event": "cache_backward_lookup_hit",
                        "business_type": business_type,
                        "typed_key": cache_key,
                        "legacy_key": base_key,
                        "result_count": len(legacy_results),
                    },
                )
                # Hydrate the typed cache to self-heal going forward
                try:
                    save_to_cache(legacy_results, cache_key, self.config.cache_dir)
                except Exception as e:
                    logger.warning(
                        f"Failed to hydrate typed cache '{cache_key}' from legacy key '{base_key}': {e}"
                    )
                return legacy_results
        
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
        classified_data = []
        for item in pharmacies:
            try:
                with credit_tracker.track_usage('perplexity'):
                    pharmacy = PharmacyData.from_dict(item)
                    classification_result = self.classifier.classify_pharmacy(pharmacy)
                    item['classification'] = classification_result.to_dict()
            except Exception as e:
                logger.warning(f"Failed to classify pharmacy {item.get('id', 'N/A')}: {e}")
                item['classification_error'] = str(e)
            classified_data.append(item)  # Add item to classified_data regardless of success/failure
        return classified_data
    
    def _verify_pharmacies(self, pharmacies: List[Dict]) -> List[Dict]:
        """Verify pharmacy information using Google Places API.
        
        This method handles exceptions during verification by adding a 'verification_error'
        field to the pharmacy data rather than failing the entire process.
        
        Args:
            pharmacies: List of pharmacy records to verify
            
        Returns:
            List of pharmacy records with verification results or error information
        """
        verified_data = []
        for item in pharmacies:
            try:
                with credit_tracker.track_usage('google_places'):
                    verification = verify_pharmacy(item)
                    item['verification'] = verification
            except Exception as e:
                logger.warning(f"Failed to verify pharmacy {item.get('id', 'N/A')}: {e}")
                item['verification_error'] = str(e)
            verified_data.append(item)  # Add item to verified_data regardless of success/failure
        return verified_data
    
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

        # Project schema v2 fields onto each record (missing -> None)
        projected: List[Dict] = []
        # Determine extra gated fields
        include_country_code = 0
        try:
            include_country_code = int(getattr(self.config, "INTERNATIONAL_ENABLED", 0))
        except Exception:
            include_country_code = 0
        gated_fields = ["country_code"] if include_country_code else []
        for item in pharmacies:
            # copy to avoid mutating input reference
            rec = dict(item)
            # Normalize address if raw address present; do not overwrite preset normalized fields
            try:
                addr_raw = rec.get("address")
                if addr_raw:
                    addr_norm = normalize_address(addr_raw, config=self.config)
                    for k in [
                        "address_line1",
                        "address_line2",
                        "city",
                        "state",
                        "postal_code",
                        "country_iso2",
                        "country_code",  # may be present when intl enabled
                    ]:
                        if k in addr_norm and addr_norm.get(k) is not None and rec.get(k) in (None, ""):
                            rec[k] = addr_norm[k]
            except Exception as e:
                logger.debug(f"Address normalization failed for {rec.get('id', 'N/A')}: {e}")
            
            # Normalize phone if raw phone present; do not overwrite preset normalized fields
            try:
                phone_raw = rec.get("phone")
                if phone_raw:
                    phone_norm = normalize_phone(phone_raw, config=self.config)
                    for k in ["phone_e164", "phone_national"]:
                        if k in phone_norm and phone_norm.get(k) is not None and rec.get(k) in (None, ""):
                            rec[k] = phone_norm[k]
            except Exception as e:
                logger.debug(f"Phone normalization failed for {rec.get('id', 'N/A')}: {e}")
            
            # Enrich contact fields from NPI data if available
            try:
                npi_data = rec.get("npi_data")
                if npi_data:
                    existing_contact = {
                        "contact_name": rec.get("contact_name"),
                        "contact_role": rec.get("contact_role"),
                        "contact_source": rec.get("contact_source")
                    }
                    contact_enriched = enrich_contact_from_npi(npi_data, existing_contact=existing_contact)
                    for k in ["contact_name", "contact_role", "contact_source"]:
                        if contact_enriched.get(k) is not None:
                            rec[k] = contact_enriched[k]
            except Exception as e:
                logger.debug(f"NPI contact enrichment failed for {rec.get('id', 'N/A')}: {e}")

            # Enforce gating: remove country_code from record when international disabled
            if not include_country_code and "country_code" in rec:
                rec.pop("country_code", None)
            for field_name in SCHEMA_V2_BASE_FIELDS + SCHEMA_V2_REMAINING_FIELDS + gated_fields:
                rec.setdefault(field_name, None)
            projected.append(rec)

        # Save as JSON
        with open(json_output_file, 'w') as f:
            json.dump(projected, f, indent=2)
        
        # Save as CSV
        df = pd.DataFrame(projected)
        # Ensure new fields are appended at the end to keep backward compatibility
        base_cols = [c for c in df.columns if c not in SCHEMA_V2_BASE_FIELDS + SCHEMA_V2_REMAINING_FIELDS + ["country_code"]]
        # Column ordering policy:
        # - INTERNATIONAL_ENABLED=0: base_cols -> remaining_fields -> base_fields
        #   (so the last N columns are exactly base_fields)
        # - INTERNATIONAL_ENABLED=1: base_cols -> base_fields -> remaining_fields -> country_code
        if include_country_code:
            final_cols = base_cols
            final_cols += [c for c in SCHEMA_V2_BASE_FIELDS if c in df.columns]
            final_cols += [c for c in SCHEMA_V2_REMAINING_FIELDS if c in df.columns]
            if "country_code" in df.columns:
                final_cols += ["country_code"]
        else:
            final_cols = base_cols
            final_cols += [c for c in SCHEMA_V2_REMAINING_FIELDS if c in df.columns]
            final_cols += [c for c in SCHEMA_V2_BASE_FIELDS if c in df.columns]
        df = df.reindex(columns=final_cols)
        df.to_csv(csv_output_file, index=False)
        
        logger.info(f"Saved {len(pharmacies)} pharmacies to {output_dir}")
        return json_output_file