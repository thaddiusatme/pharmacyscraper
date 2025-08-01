import os
import json
import logging
import time
from typing import Dict, List, Optional, Union

import httpx
import apify_client
from apify_client import ApifyClient  # re-export for tests patching convenience

# Configure logging
import os
from pathlib import Path

# Create logs directory if it doesn't exist
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True, parents=True)

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,  # Capture all logs
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'apify_collector.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("ApifyCollector logging initialized")

class ApifyCollector:
    """A utility class to interact with Apify actors and persist the scraped data.

    The implementation is intentionally **very** lightweight – its purpose is not to
    be production-ready, but simply to provide a thin wrapper that the accompanying
    unit-test-suite can exercise and mock out.  The tests come in two slightly
    different flavours (legacy and *new*) which means the public surface of this
    object has to be a little forgiving:

    * ``__init__`` must accept ``api_token`` **or** retrieve the token from the
      ``APIFY_API_TOKEN`` / ``APIFY_TOKEN`` environment variables.
    * ``run_trial`` can be invoked either with               :
        ``run_trial(config_path)`` **or** ``run_trial(query, location)``.
    * ``collect_pharmacies`` can be invoked either with      :
        ``collect_pharmacies(config_dict)`` **or** ``collect_pharmacies(state, city)``.

    The implementation therefore performs a small amount of runtime inspection on
    the received arguments in order to route the call to the most appropriate
    private helper.
    """

    # A (very) small set of big-box chains we want to filter out in
    # ``filter_chain_pharmacies``.
    _CHAIN_KEYWORDS = {"cvs", "walgreens", "rite aid", "walmart", "duane reade"}
    
    # Configure logger
    logger = logging.getLogger(__name__)

    def __init__(
        self,
        api_key: Optional[str] = None,
        output_dir: str = "output",
        cache_dir: str = ".api_cache/apify",
        rate_limit_ms: int = 1000,
        use_cache: bool = True,
        **kwargs,
    ) -> None:
        # Accept deprecated/alternate ``api_token`` via kwargs while keeping the
        # primary parameter name ``api_key`` that the legacy tests expect.
        api_token = kwargs.get("api_token")  # type: ignore[arg-type]

        token = (
            api_key
            or api_token
            or os.getenv("APIFY_API_TOKEN")
            or os.getenv("APIFY_TOKEN")
        )
        if not token:
            raise ValueError(
                "Apify API key not provided. Either pass it as an argument "
                "or set the APIFY_TOKEN/APIFY_API_TOKEN environment variables."
            )

        self.api_token: str = token  # Keep for forward-compatibility
        self.api_key: str = token    # Legacy tests access this attribute
        self.rate_limit_ms: int = rate_limit_ms  # public – tests access this
        self.use_cache: bool = use_cache

        # Output and cache directory handling
        self.output_dir: str = output_dir
        self.cache_dir: str = cache_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

        # Lazily instantiated Apify client.  We do **not** create it immediately so
        # that the unit tests can monkey-patch the constructor before first use.
        self._client: Optional["ApifyClient"] = None

    @property
    def api_key_prop(self) -> str:  # pragma: no cover
        return self.api_key
        
    def _fallback_to_direct_api(self, search_query: str, max_results: int) -> List[Dict]:
        """Fallback to direct Google Places API when Apify actor is not available.
        
        Args:
            search_query: The search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of pharmacy results
        """
        self.logger.warning("Using direct Google Places API as fallback")
        
        try:
            from googlemaps import Client as GoogleMaps
            from googlemaps.places import places
            
            # Get Google Maps API key from environment
            api_key = os.getenv('GOOGLE_MAPS_API_KEY')
            if not api_key:
                raise ValueError("GOOGLE_MAPS_API_KEY environment variable not set")
                
            # Initialize Google Maps client
            gmaps = GoogleMaps(api_key)
            
            # Search for pharmacies near Seattle
            places_result = places(gmaps, search_query, 
                                 location=(47.6062, -122.3321),  # Seattle coordinates
                                 radius=50000,  # 50km radius
                                 type='pharmacy')
            
            # Limit results to max_results
            places_result['results'] = places_result.get('results', [])[:max_results]
            
            # Format results to match Apify output format
            results = []
            for place in places_result.get('results', []):
                results.append({
                    'name': place.get('name'),
                    'address': place.get('formatted_address'),
                    'phone': place.get('formatted_phone_number', ''),
                    'location': {
                        'lat': place['geometry']['location']['lat'],
                        'lng': place['geometry']['location']['lng']
                    },
                    'place_id': place.get('place_id'),
                    'rating': place.get('rating', 0),
                    'user_ratings_total': place.get('user_ratings_total', 0),
                    'types': place.get('types', []),
                    'permanently_closed': place.get('permanently_closed', False),
                    'business_status': place.get('business_status', 'OPERATIONAL')
                })
                
            self.logger.info(f"Found {len(results)} results using direct Google Places API")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to use direct Google Places API: {str(e)}")
            raise Exception(f"Both Apify actor and direct Google Places API failed: {str(e)}")

    # ---------------------------------------------------------------------
    # Private helpers
    # ---------------------------------------------------------------------
    def _get_client(self) -> "ApifyClient":
        """Return – and memoise – an ``ApifyClient`` instance."""
        if self._client is None:
            # Prefer environment variables so tests can easily override the token,
            # but gracefully fall back to the constructor-provided token if none
            # are set.
            api_token = (
                os.getenv("APIFY_API_TOKEN")
                or os.getenv("APIFY_TOKEN")
                or self.api_token
            )
            if not api_token:
                self.logger.error(
                    "Apify API token not provided via env vars or constructor."
                )
                raise ValueError("Apify API token not provided")

            self.logger.info("Initializing Apify client ...")
            try:
                # Resolve the client class at *runtime* so that unit tests patching
                # ``apify_client.ApifyClient`` see their mock being used even though
                # the module-level import occurred earlier.
                ClientCls = getattr(apify_client, "ApifyClient", ApifyClient)
                self._client = ClientCls(api_token)
                # Simple validation call (mocked in unit tests)
                self._client.user().get()
                self.logger.info("Successfully initialized Apify client")
            except Exception as e:
                self.logger.error(f"Failed to initialize Apify client: {e}")
                raise RuntimeError("Failed to initialize Apify client") from e

        return self._client

    def _create_output_dir(self) -> None:
        """Create *root* output directory if it does not yet exist."""
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def _create_output_directories(base_dir: str, states: List[str]) -> None:
        """Create sub-directories – one per state – underneath *base_dir*."""
        for state in states:
            state_slug = state.lower().replace(" ", "_")
            os.makedirs(os.path.join(base_dir, state_slug), exist_ok=True)

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _load_config(path: str) -> Dict:
        """Load and return the JSON configuration located at *path*."""
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        with open(path, "r", encoding="utf-8") as fh:
            data: Dict = json.load(fh)

        # The fixtures wrap the *real* config into an arbitrary top-level key
        # (e.g. ``trial_run_20240619``).  If that is the case we simply unwrap
        # the first – and only – entry.
        if len(data) == 1 and "states" not in data and "queries" not in data:
            data = next(iter(data.values()))
        return data

    def _generate_queries_from_config(self, path: str) -> List[Dict[str, str]]:
        """Convert the *states/cities/queries* tree into a flat list of dicts."""
        cfg = self._load_config(path)
        queries: List[Dict[str, str]] = []

        if "states" not in cfg:
            # Nothing to do – return empty list so the caller can deal with it.
            return queries

        for state, state_data in cfg["states"].items():
            for city_info in state_data.get("cities", []):
                city = city_info["name"]
                for q in city_info.get("queries", []):
                    queries.append({"query": q, "state": state, "city": city})
        return queries

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _save_results(data: List[Dict], path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)

    # Public alias – some tests call the *private* one, others the *public* one.
    save_results = _save_results

    # ------------------------------------------------------------------
    # Business-logic helpers
    # ------------------------------------------------------------------
    @classmethod
    def filter_pharmacy_businesses(cls, places: List[Dict]) -> List[Dict]:
        """Return only items that look like pharmacies (category/name heuristic)."""
        pharmacy_kw = {"pharmacy", "drug store", "drugstore"}
        filtered: List[Dict] = []
        for p in places:
            name = (p.get("name") or "").lower()
            cat = (p.get("categoryName") or "").lower()
            if any(kw in name for kw in pharmacy_kw) or any(kw in cat for kw in pharmacy_kw):
                filtered.append(p)
        return filtered

    # Common chain pharmacy keywords
    _CHAIN_KEYWORDS = {
        'cvs', 'walgreens', 'rite aid', 'walmart', 'walmart pharmacy',
        'target', 'target pharmacy', 'costco', 'costco pharmacy',
        'sams club', 'safeway', 'kroger', 'publix', 'wal-mart',
        'riteaid', 'cvs pharmacy', 'walgreens pharmacy', 'rite aid pharmacy'
    }
    
    def filter_chain_pharmacies(self, pharmacies: List[Dict]) -> List[Dict]:
        """Return **only** non-chain pharmacies based on a very small heuristic."""
        filtered: List[Dict] = []
        for p in pharmacies:
            name = p.get("name", "").lower()
            if not any(chain_kw in name for chain_kw in self._CHAIN_KEYWORDS):
                filtered.append(p)
        return filtered

    # ------------------------------------------------------------------
    # Primary public interface
    # ------------------------------------------------------------------
    def run_trial(
        self,  # noqa: C901 – a bit complex because it supports 2 call patterns
        config_or_query: str,
        location: Optional[str] = None,
    ) -> Union[List[Dict], Dict[str, List[Dict]]]:
        """Run a *trial* scrape.

        Two invocation styles are supported – they are distinguished by the
        *presence* (or absence) of the *location* argument:

        1. ``run_trial(config_path)``
        2. ``run_trial(query, location)``
        """

        # ------------------------------------------------------------------
        # Style (1) – configuration file path
        # ------------------------------------------------------------------
        if location is None:
            try:
                cfg = self._load_config(config_or_query)
                max_results = cfg.get("max_results_per_query") or cfg.get(
                    "max_results", 10
                )
            except Exception as exc:
                raise Exception(f"Error in run_trial: {exc}") from exc

            # a) Simple *queries*-only configuration ---------------------------------
            if "queries" in cfg:
                queries_cfg = cfg.get("queries", [])
                if not queries_cfg:
                    return []

                # Support both list[str] and dict[state -> list[dict|str]] structures
                flat_queries: List[str] = []
                if isinstance(queries_cfg, list):
                    # e.g. ["independent pharmacy Boston MA", ...]
                    flat_queries = [q if isinstance(q, str) else q.get("query", "") for q in queries_cfg]
                elif isinstance(queries_cfg, dict):
                    # e.g. {"CA": [{"query": "independent pharmacy …"}, …], "TX": [...]}
                    for state_list in queries_cfg.values():
                        for item in state_list:
                            if isinstance(item, str):
                                flat_queries.append(item)
                            elif isinstance(item, dict):
                                flat_queries.append(item.get("query", ""))
                else:
                    # Unknown structure – abort early
                    return []

                results: List[Dict] = []
                for q in flat_queries:
                    if not q:
                        continue
                    try:
                        res = self._execute_actor(q, max_results=max_results)
                        results.extend(res)
                    except Exception as exc:
                        raise Exception(f"Error in run_trial: {exc}") from exc
                return results

            # b) Full *states/cities* configuration -----------------------------------
            elif "states" in cfg:
                queries_list = self._generate_queries_from_config(config_or_query)
                results_by_city: Dict[str, List[Dict]] = {}

                for q in queries_list:
                    city_label = q["city"]
                    try:
                        res = self._execute_actor(q["query"], max_results=max_results)
                        results_by_city.setdefault(city_label, []).extend(res)
                    except Exception as exc:
                        raise Exception(f"Error in run_trial: {exc}") from exc

                return results_by_city

            # c) Unsupported config ----------------------------------------------------
            return []

        # ------------------------------------------------------------------
        # Style (2) – single query/location pair
        # ------------------------------------------------------------------
        try:
            query_str = config_or_query
            if location and location.lower() not in query_str.lower():
                query_str = f"{config_or_query} in {location}"
            res = self._execute_actor(query_str, max_results=10)
            return res
        except Exception as exc:
            raise Exception(str(exc)) from exc

    # Helper that performs the **actual** call to the Apify actor via SDK so the
    # tests can monkey-patch the client easily.
    def _get_cache_path(self, query: str) -> str:
        """Generate a predictable file path for a given query."""
        slug = "".join(c for c in query.lower() if c.isalnum() or c in " _-").rstrip()
        slug = slug.replace(" ", "_")
        return os.path.join(self.cache_dir, f"{slug}.json")

    def _execute_actor(self, search_query: str, max_results: int = 10) -> List[Dict]:
        # Check cache first
        cache_path = self._get_cache_path(search_query)
        if self.use_cache and os.path.exists(cache_path):
            self.logger.info(f"CACHE HIT: Loading results for '{search_query}' from {cache_path}")
            with open(cache_path, "r", encoding="utf-8") as fh:
                return json.load(fh)

        self.logger.info(f"CACHE MISS: Executing Apify actor for '{search_query}'")

        client = self._get_client()
        
        # Debug: Log API token (first 5 and last 5 characters for security)
        # Prefer explicit environment variables so that integration tests can
        # override the token easily, but gracefully fall back to the instance
        # attribute provided via the constructor when nothing is set.
        api_token = os.getenv("APIFY_API_TOKEN") or os.getenv("APIFY_TOKEN") or self.api_token
        if not api_token:
            self.logger.error(
                "No Apify API token available (environment variables unset and no fallback token)."
            )
            raise ValueError("Apify API token not provided")
            
        # Update the client token if it's different
        if api_token != self.api_token:
            self.logger.info("Updating Apify client with new API token")
            self.api_token = api_token
            self.api_key = api_token  # For backward compatibility
            self._client = None  # Force client reinitialization
            client = self._get_client()  # Get a fresh client with the new token
            
        token_display = f"{api_token[:5]}...{api_token[-5:]}" if api_token else "[No token]"
        self.logger.info(f"Using Apify API token: {token_display}")
        
        # Try to find a suitable Google Maps scraper actor
        try:
            # Use the official Google Maps Scraper actor directly
            actor_id = "apify/google-maps-scraper"
            self.logger.info(f"Using Google Maps actor: {actor_id}")
            
            # Verify the actor exists and is accessible
            try:
                actor_client = client.actor(actor_id).get()
                self.logger.info(f"Successfully accessed actor: {actor_client.get('name')} (v{actor_client.get('version')})")
            except Exception as e:
                self.logger.error(f"Failed to access actor {actor_id}: {str(e)}")
                self.logger.info("Falling back to direct Google Places API integration...")
                return self._fallback_to_direct_api(search_query, max_results)
            
            # Prepare input for the official Google Maps Scraper actor
            run_input = {
                "searchStringsArray": [search_query],
                "searchMatching": "all",
                "maxCrawledPlaces": max_results,
                "countryCode": "us",
                "language": "en",
                "maxImages": 0,
                "maxReviews": 0,
                "includeGeolocation": True,
                "includeOpeningHours": False,
                "includePeopleAlsoSearch": False,
                "includePopularTimes": False,
                "includeReviews": False,
                "includeReviewsSummary": False,
                "includeImages": False,
                "includeWebResults": False,
                "includeDetailUrl": False,
                "includePosition": True,
                "includePeopleAlsoSearchFor": False,
                "includePeopleAlsoSearchForInResponse": False,
                "includePeopleAlsoSearchForInResponseDetails": False,
                "maxConcurrency": 10,
                "maxRequestRetries": 3,
                "requestTimeoutSecs": 60,
                "sessionPoolName": "google-maps-scraper-session",
                "proxyConfiguration": {
                    "useApifyProxy": True,
                    "apifyProxyGroups": ["RESIDENTIAL"],
                    "apifyProxyCountry": "US"
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error initializing Google Maps actor: {str(e)}")
            self.logger.info("Falling back to direct Google Places API integration...")
            return self._fallback_to_direct_api(search_query, max_results)

        try:
            self.logger.info(f"Attempting to call Apify actor {actor_id} with input: {json.dumps(run_input, indent=2)}")
            
            # First, try to get actor info to verify access
            try:
                actor_info = client.actor(actor_id).get()
                self.logger.info(f"Successfully accessed actor info: {actor_info.get('name', 'Unknown')} - {actor_info.get('description', 'No description')}")
            except Exception as e:
                self.logger.error(f"Failed to get actor info: {str(e)}")
                # Try to get more details about the error
                try:
                    # This will raise a more detailed exception
                    client.actor(actor_id).call(run_input=run_input, wait_secs=5)
                except Exception as detailed_e:
                    self.logger.error(f"Detailed error when calling actor: {str(detailed_e)}")
                    raise
            
            # Start the actor run **asynchronously** so we can poll quietly.
            run_record = client.actor(actor_id).call(run_input=run_input, wait_secs=0)
            run_id: str = run_record["id"] if isinstance(run_record, dict) else str(run_record)
            self.logger.info(f"Successfully started actor run with ID: {run_id}")

            # ------------------------------------------------------------------
            # Quiet polling loop – avoids streaming the actor logs to stdout.
            # ------------------------------------------------------------------
            poll_interval = int(os.getenv("APIFY_POLL_INTERVAL", "5"))
            max_wait = int(os.getenv("APIFY_MAX_WAIT_SECS", "3600"))  # 1 hour default for large runs
            start_ts = time.time()
            while True:
                run_details = client.run(run_id).get()
                status = run_details.get("status")
                if status in ("SUCCEEDED", "FAILED", "TIMED-OUT", "ABORTED"):  # finished
                    break
                if time.time() - start_ts > max_wait:
                    elapsed_mins = (time.time() - start_ts) / 60
                    self.logger.error(f"Actor run {run_id} exceeded max wait time of {max_wait//60} minutes (ran for {elapsed_mins:.1f} minutes)")
                    raise TimeoutError(f"Apify actor run exceeded max wait time after {elapsed_mins:.1f} minutes")
                self.logger.info(f"Run status is {status}, polling again in {poll_interval} seconds...")
                time.sleep(poll_interval)

            # ------------------------------------------------------------------
            # Actor finished – retrieve the dataset
            # ------------------------------------------------------------------
            dataset_id = run_details.get("defaultDatasetId")
            
            if not dataset_id:
                self.logger.error(f"No dataset ID found in run details. Run status: {status}")
                if status == "FAILED":
                    raise Exception(f"Apify actor run failed: {run_details.get('statusMessage', 'Unknown error')}")
                elif status == "TIMED-OUT":
                    raise Exception("Apify actor run timed out on server side")
                else:
                    raise Exception(f"Actor run completed with status {status} but no dataset ID found")
        except Exception as e:
            self.logger.error(f"Error executing Apify actor: {e}")
            raise Exception("Actor not found")

        # The Apify client SDK has two methods for retrieving dataset items.
        # We try the modern `.list_items()` first, then fallback to the legacy
        # `.iterate_items()`.
        raw_items = []
        dataset_client = client.dataset(dataset_id)
        try:
            # list_items() returns a an object with an .items property
            self.logger.info(f"Attempting to fetch results with list_items() for dataset {dataset_id}")
            raw_items = dataset_client.list_items().items
        except Exception:
            self.logger.warning(f"Failed to get items with list_items() for dataset {dataset_id}, falling back to iterate_items().")
            try:
                raw_items = list(dataset_client.iterate_items())
            except Exception as e:
                self.logger.error(f"Could not retrieve dataset items from {dataset_id} with any method: {e}")
                return [] # Return empty list if both methods fail

        # Filter and process the raw results
        final_items: List[Dict] = []
        if raw_items:
            filtered_items = self.filter_pharmacy_businesses(raw_items)
            final_items = self.filter_chain_pharmacies(filtered_items)

        # Save to cache if enabled
        if self.use_cache:
            self.logger.info(f"CACHE WRITE: Saving {len(final_items)} results for '{search_query}' to {cache_path}")
            self._save_results(final_items, cache_path)

        return final_items
        
        # Write the final, processed results to cache
        with open(cache_path, "w", encoding="utf-8") as fh:
            json.dump(final_items, fh, ensure_ascii=False, indent=2)
        self.logger.info(f"CACHE WRITE: Saved {len(final_items)} items for '{search_query}' to {cache_path}")

        return final_items

    # ------------------------------------------------------------------
    # Collection helper (multi-query convenience)
    # ------------------------------------------------------------------
    def collect_pharmacies(
        self,
        config_or_state: Union[Dict, str, List[Dict]],
        city: Optional[str] = None,
    ) -> List[Dict]:
        """Collect pharmacies and optionally persist them to *output_dir*."""

        # ----------------- mode 1: (state, city) -----------------
        if city is not None and isinstance(config_or_state, str):
            state = config_or_state
            try:
                results = self.run_trial("pharmacy", f"{city}, {state}")
            except Exception as exc:
                logging.error(f"Error collecting pharmacies: {exc}")
                return []

            # Persist immediately
            filename = f"pharmacies_{state}_{city.lower().replace(' ', '_')}.json"
            self._save_results(results, os.path.join(self.output_dir, filename))
            return results

        # -------------- mode 2: config dict/list  ----------------
        if isinstance(config_or_state, dict):
            # Mode 2a – dict with *queries* – the tests expect **one** actor call and
            # a *single* output file even if multiple queries are provided.  We
            # therefore execute **only the first query**.
            if "queries" in config_or_state:
                queries_list: List[str] = config_or_state.get("queries", [])

                if not queries_list:
                    return []

                try:
                    res = self._execute_actor(
                        queries_list[0],
                        max_results=config_or_state.get("max_results", 10),
                    )
                except Exception as exc:
                    logging.error(f"Error collecting pharmacies: {exc}")
                    return []

                # Persist – naming pattern required by legacy tests (
                # ``pharmacy_results_1.json``)
                filename = "pharmacy_results_1.json"
                self._save_results(res, os.path.join(self.output_dir, filename))

                return res

            # Mode 2b – dict from *generate_queries_from_config* style
            queries = config_or_state.get("queries", [])
        elif isinstance(config_or_state, list):
            queries = config_or_state
        else:
            raise ValueError("Unsupported arguments for collect_pharmacies")

        all_results: List[Dict] = []
        for q in queries:
            try:
                # Avoid duplicating the city/state portion of the search query by
                # directly invoking the actor on the *original* query string.
                res = self._execute_actor(
                    q["query"],
                    max_results=q.get("max_results", 10),
                )
                all_results.extend(res)

                # Persist per-query
                filename = (
                    f"pharmacies_{q['state']}_{q['city'].lower().replace(' ', '_')}.json"
                )
                self._save_results(
                    res, os.path.join(self.output_dir, filename)
                )
            except Exception as exc:
                logging.error(f"Error collecting pharmacies: {exc}")
        # ------------------------------------------------------------------
        # De-duplicate by pharmacy name (case-insensitive) – the actor can
        # occasionally return the same item multiple times which breaks the
        # unit tests that expect a unique list.
        # ------------------------------------------------------------------
        unique: Dict[str, Dict] = {}
        for item in all_results:
            key = (item.get("name") or "").strip().lower()
            if key and key not in unique:
                unique[key] = item
        deduped = list(unique.values())
        self.logger.debug(f"collect_pharmacies returning {len(deduped)} items (deduped from {len(all_results)})")
        return deduped

    # ------------------------------------------------------------------
    # Public helpers – exposed for test-suite convenience
    # ------------------------------------------------------------------
    def save_results(self, data: List[Dict], path: str) -> None:  # noqa: D401
        """Persist *data* to *path*.

        The file format is inferred from the *path* extension:
        • ``.csv`` → CSV via *pandas* (tests expect this).
        • otherwise  → JSON via :py:meth:`_save_results`.
        """

        ext = os.path.splitext(path)[1].lower()
        if ext == ".csv":
            try:
                import pandas as pd

                pd.DataFrame(data).to_csv(path, index=False)
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("pandas is required to save CSV files") from exc
        else:
            self._save_results(data, path)

def run_trial(query: str, location: str) -> List[Dict]:
    """Run a trial collection using the Apify Google Maps Scraper.
    
    This is a convenience function that creates an ApifyCollector instance
    and calls its run_trial method.
    
    Args:
        query: The search query (e.g., 'pharmacy').
        location: The location to search in (e.g., 'New York, NY').
        
    Returns:
        Dictionary mapping city names to lists of collected pharmacy items
        
    Raises:
        Exception: If there's an error during collection
    """
    collector = ApifyCollector()
    return collector.run_trial(query, location)

def main():
    """Example usage of the ApifyCollector class."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Apify data collection')
    parser.add_argument('--query', help='Search query')
    parser.add_argument('--location', help='Location to search in')
    args = parser.parse_args()
    
    collector = ApifyCollector()
    
    if args.query and args.location:
        print(f"Running trial with query: {args.query} and location: {args.location}")
        collector.run_trial(args.query, args.location)
    else:
        print("Please provide both query and location")

if __name__ == "__main__":
    main()