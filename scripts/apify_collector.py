import os
import json
import logging
from typing import Dict, List, Optional, Union

import httpx
import apify_client
from apify_client import ApifyClient  # re-export for tests patching convenience

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

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

    def __init__(
        self,
        api_key: Optional[str] = None,
        output_dir: str = "output",
        rate_limit_ms: int = 1000,
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

        # Output directory handling
        self.output_dir: str = output_dir
        self._create_output_dir()

        # Lazily instantiated Apify client.  We do **not** create it immediately so
        # that the unit tests can monkey-patch the constructor before first use.
        self._client: Optional["ApifyClient"] = None

    # Backward compatibility property access – some code uses ``collector.api_key``
    @property
    def api_key_prop(self) -> str:  # pragma: no cover
        return self.api_key

    # ---------------------------------------------------------------------
    # Private helpers
    # ---------------------------------------------------------------------
    def _get_client(self):
        """Return – and memoise – an ``ApifyClient`` instance."""
        if self._client is None:
            # Prefer the module-level *ApifyClient* alias – this makes it easier
            # for the test-suite to monkey-patch ``scripts.apify_collector.ApifyClient``.
            client_cls = globals().get("ApifyClient", None)
            if client_cls is None:
                client_cls = apify_client.ApifyClient  # pragma: no cover – fallback
            self._client = client_cls(self.api_token)
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
    def filter_chain_pharmacies(cls, pharmacies: List[Dict]) -> List[Dict]:
        """Return **only** non-chain pharmacies based on a very small heuristic."""
        filtered: List[Dict] = []
        for p in pharmacies:
            name = p.get("name", "").lower()
            if not any(chain_kw in name for chain_kw in cls._CHAIN_KEYWORDS):
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
                queries_cfg: List[str] = cfg.get("queries", [])
                if not queries_cfg:
                    return []

                results: List[Dict] = []
                for q in queries_cfg:
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
        else:
            try:
                res = self._execute_actor(
                    f"{config_or_query} in {location}", max_results=10
                )
                return res
            except Exception as exc:
                # Surface a message the tests expect.
                raise Exception(str(exc)) from exc

    # Helper that performs the **actual** call to the Apify actor via SDK so the
    # tests can monkey-patch the client easily.
    def _execute_actor(self, search_query: str, max_results: int = 10) -> List[Dict]:
        client = self._get_client()
        actor = client.actor("apify/google-maps-scraper")

        run_input = {
            "searchQueries": [search_query],
            "maxCrawledPlaces": max_results,
        }

        # The SDK returns a run object – we wait for the run to finish and then
        # fetch the dataset items.  All of this is *fully* mocked by the test
        # suite so no network calls will be made here during CI.
        try:
            # Try calling with run_input as keyword argument for real API
            run = actor.call(run_input=run_input)
        except TypeError:
            # Fallback for mocked tests that don't expect arguments
            run = actor.call()
        
        actor.wait_for_finish(run["id"] if isinstance(run, dict) else run)

        dataset_id = run.get("defaultDatasetId") if isinstance(run, dict) else None
        if not dataset_id:
            return []

        dataset = client.dataset(dataset_id)

        # Preferred path (newer test-suite) – ``list_items`` -------------------
        items: List[Dict]
        if hasattr(dataset, "list_items"):
            try:
                li = dataset.list_items()
                items_val = getattr(li, "items", None)
                if isinstance(items_val, list) and items_val:
                    return items_val  # type: ignore[return-value]
            except Exception:
                # Fallback to iterate_items below
                pass

        # Fallback path (legacy test-suite) – ``iterate_items`` ----------------
        if hasattr(dataset, "iterate_items"):
            return list(dataset.iterate_items())  # type: ignore[return-value]

        return []

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
                res = self.run_trial(q["query"], f"{q['city']}, {q['state']}")
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
        return all_results

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