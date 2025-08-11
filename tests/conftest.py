import pytest
import sys
from pathlib import Path
# Ensure project 'src' dir is on PYTHONPATH for local test execution
root_dir = Path(__file__).resolve().parents[1]
src_path = root_dir / "src"
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Provide lightweight stubs for heavy third-party libs that aren't needed for unit tests
from types import ModuleType
from unittest.mock import MagicMock as _MM

# -- pandas stub (already handled above) --
# Ensure stubs for other external deps the project imports during import time
def _ensure_stub(name: str, attrs: dict | None = None):
    if name in sys.modules:
        return
    stub = ModuleType(name)
    if attrs:
        for k, v in attrs.items():
            setattr(stub, k, v)
    sys.modules[name] = stub

# _ensure_stub("pandas", {"DataFrame": _MM(), "Series": _MM()})  # Commented out - use real pandas
_ensure_stub("httpx", {"Client": _MM(), "request": _MM()})
_ensure_stub("apify_client", {"ApifyClient": _MM()})
# The package imports 'from apify_client import ApifyClient'
_ensure_stub("apify_client.ApifyClient", {})
_ensure_stub("cachetools", {"TTLCache": _MM()})

# tenacity stub with common helpers & RetryError
class _FakeRetryError(Exception):
    pass
_ensure_stub("tenacity", {
    "retry": lambda *a, **k: (lambda f: f),
    "stop_after_attempt": lambda *a, **k: None,
    "wait_exponential": lambda *a, **k: None,
    "retry_if_exception_type": lambda *a, **k: None,
    "retry_if_exception": lambda *a, **k: None,
    "before_sleep_log": lambda *a, **k: None,
    "RetryError": _FakeRetryError,
})

# googlemaps has nested submodule access, so add a thin stub
gm_stub = ModuleType("googlemaps")
setattr(gm_stub, "Client", _MM())
setattr(gm_stub, "places", _MM())
sys.modules["googlemaps"] = gm_stub

# Pandas stub only when truly missing - removed to allow real pandas to be used
# The real pandas package is now available in the virtual environment
try:
    import pandas as _pd  # noqa: F401
    # Real pandas is available, don't override it
except ModuleNotFoundError:
    # Only create stub if pandas is truly missing
    from types import ModuleType
    from unittest.mock import MagicMock as _MM
    _pandas_stub = ModuleType("pandas")
    _pandas_stub.DataFrame = _MM()
    _pandas_stub.Series = _MM()
    _pandas_stub.concat = _MM()  # Add concat for completeness
    sys.modules["pandas"] = _pandas_stub
from unittest.mock import patch, MagicMock
from types import SimpleNamespace
import json
from pathlib import Path

from pharmacy_scraper.utils.api_usage_tracker import credit_tracker

@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary config file for testing."""
    config_data = {
        "api_keys": {
            "apify": "test_api_key",
            "google_places": "test_google_key",
            "perplexity": "test_perplexity_key"
        },
        "locations": [
            {
                "state": "CA",
                "cities": ["Test City"],
                "queries": ["test query"]
            }
        ],
        "max_results_per_query": 5,
        "output_dir": str(tmp_path),
        "cache_dir": str(tmp_path / "cache"),
        "verify_places": True
    }
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config_data))
    return str(config_file)

@pytest.fixture
def orchestrator_fixture(temp_config_file):
    """Fixture for PipelineOrchestrator with mocked dependencies."""
    from pharmacy_scraper.orchestrator.pipeline_orchestrator import PipelineOrchestrator, PipelineConfig
    
    with patch('pharmacy_scraper.orchestrator.pipeline_orchestrator.ApifyCollector') as mock_apify_collector, \
         patch('pharmacy_scraper.orchestrator.pipeline_orchestrator.remove_duplicates') as mock_remove_duplicates, \
         patch('pharmacy_scraper.orchestrator.pipeline_orchestrator.verify_pharmacy') as mock_verify_pharmacy, \
         patch('pharmacy_scraper.orchestrator.pipeline_orchestrator.load_from_cache') as mock_load_from_cache, \
         patch('pharmacy_scraper.orchestrator.pipeline_orchestrator.save_to_cache') as mock_save_to_cache:

        # Create a mock Classifier
        mock_classifier = MagicMock()
        # Patch the Classifier class to return our mock when instantiated
        with patch('pharmacy_scraper.orchestrator.pipeline_orchestrator.Classifier', return_value=mock_classifier):
            
            mock_remove_duplicates.side_effect = lambda df, **kwargs: df
            mock_load_from_cache.return_value = None
            
            orchestrator = PipelineOrchestrator(temp_config_file)
            
            # Group mocks for easy access
            mocks = SimpleNamespace(
                apify=orchestrator.collector,
                classifier=mock_classifier,  # This is now the instance mock, not the class mock
                remove_duplicates=mock_remove_duplicates,
                verify_pharmacy=mock_verify_pharmacy,
                cache_load=mock_load_from_cache,
                cache_save=mock_save_to_cache
            )
            
            yield SimpleNamespace(
                orchestrator=orchestrator,
                mocks=mocks
            )

@pytest.fixture(autouse=True)
def reset_credit_tracker():
    """Reset the credit tracker singleton before each test to ensure isolation."""
    credit_tracker.reset()