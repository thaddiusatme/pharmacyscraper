# Testing Guide

This document provides detailed information about the test suite and how to run tests for the Independent Pharmacy Verification project.

## Table of Contents
- [Test Structure](#test-structure)
- [Running Tests](#running-tests)
- [Test Coverage](#test-coverage)
- [Writing Tests](#writing-tests)
- [Testing Apify Integration](#testing-apify-integration)
- [Testing Credit Management](#testing-credit-management)
- [Mocking External Services](#mocking-external-services)
- [Continuous Integration](#continuous-integration)

## Test Structure

The test suite is organized as follows:

```
tests/
├── __init__.py                 # Makes test directory a Python package
├── conftest.py                 # Pytest fixtures and configuration
├── test_apify_integration.py    # Tests for Apify integration
├── test_classification.py       # Tests for pharmacy classification
└── test_dedup_self_heal.py     # Tests for deduplication and self-healing
```

## Running Tests

### Basic Test Execution

Run all tests:
```bash
pytest tests/
```

Run a specific test file:
```bash
pytest tests/test_apify_integration.py
```

Run a specific test function:
```bash
pytest tests/test_apify_integration.py::test_scrape_pharmacies_basic
```

### Running with Coverage

Generate a coverage report:
```bash
pytest --cov=src tests/
```

Generate HTML coverage report:
```bash
pytest --cov=src --cov-report=html tests/
open htmlcov/index.html  # View report in browser
```

## Testing Apify Integration

The Apify integration tests verify that the Google Maps Scraper works correctly with proper error handling and rate limiting.

### Live API Testing

To test against the live Apify API (requires API token):

```bash
# Set your API token
export APIFY_API_TOKEN=your_token_here

# Run the test script with specific parameters
python scripts/test_apify_integration.py \
    --state CA \
    --city "San Francisco" \
    --max-results 5
```

### Mocked Tests

Most tests use mocks to avoid hitting the real API. These are located in:
- `tests/test_apify_integration.py`
- `tests/conftest.py` (fixtures)

Example of testing with mocks:

```python
def test_scrape_pharmacies_mocked(mock_apify_client):
    """Test that scrape_pharmacies returns expected results with mocked client."""
    scraper = ApifyPharmacyScraper(api_token="test_token")
    results = scraper.scrape_pharmacies("CA", "San Francisco", max_results=3)
    
    assert len(results) == 3
    assert all(isinstance(r, dict) for r in results)
```

## Testing Credit Management

The credit management system tracks API usage and enforces limits.

### Test Files
- `tests/test_credit_tracking.py`
- `src/utils/api_usage_tracker.py`

### Example Test

```python
def test_credit_tracking():
    """Test credit tracking functionality."""
    # Reset tracker for test
    tracker = ApiUsageTracker()
    tracker.reset()
    
    # Record some usage
    tracker.record_usage(1.0, "Test API call")
    
    # Check remaining credits
    assert tracker.get_usage_summary()["remaining"] == 99.0
    
    # Test daily limit
    tracker.record_usage(24.0, "Large API call")
    with pytest.raises(CreditLimitExceededError):
        tracker.record_usage(1.0, "Should exceed daily limit")
```

### Running Credit Tests

```bash
# Run credit tracking tests
pytest tests/test_credit_tracking.py -v

# Run with coverage
pytest --cov=src.utils.api_usage_tracker tests/test_credit_tracking.py
```

## Writing Tests

### Test Naming Conventions
- Test files: `test_*.py`
- Test functions: `test_*`
- Test classes: `Test*`

### Best Practices
1. One assertion per test
2. Use descriptive test names
3. Test edge cases and error conditions
4. Use fixtures for common setup
5. Keep tests independent

### Example Test Structure

```python
def test_feature_under_test():
    # Setup
    test_data = [...]
    
    # Exercise
    result = function_under_test(test_data)
    
    # Verify
    assert result == expected_value
    
    # Cleanup (if needed)
    cleanup_resources()
```

## Mocking External Services

### Using pytest-mock

```python
def test_external_api(mocker):
    # Create a mock response
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"key": "value"}
    
    # Patch the requests.get call
    mocker.patch('requests.get', return_value=mock_response)
    
    # Test function that makes HTTP request
    result = function_that_uses_requests()
    
    assert result == expected_value
```

### Fixtures for Common Mocks

In `conftest.py`:

```python
@pytest.fixture
def mock_apify_client(mocker):
    """Fixture to mock Apify client responses."""
    mock_client = mocker.patch('apify_client.ApifyClient')
    # Configure mock client behavior
    mock_client.return_value.actor.return_value.call.return_value = {
        'defaultDatasetId': 'test-dataset-id'
    }
    mock_client.return_value.dataset.return_value.iterate_items.return_value = [
        {'title': 'Test Pharmacy', 'address': '123 Test St'}
    ]
    return mock_client
```

## Continuous Integration

GitHub Actions runs the test suite on every push and pull request. The workflow is defined in `.github/workflows/tests.yml`.

### Running CI Locally

To run the same checks that CI runs:

```bash
# Run linter
flake8 src tests

# Run type checking
mypy src tests

# Run tests with coverage
pytest --cov=src --cov-report=xml tests/
```

### Debugging CI Failures

1. Run the same commands locally that are failing in CI
2. Check the test output for specific error messages
3. Run with `-v` for verbose output
4. Use `pytest --pdb` to drop into debugger on failure

## Test Data Management

- Test data should be stored in `tests/test_data/`
- Use descriptive file names (e.g., `apify_response.json`)
- Keep test data minimal but representative
- Include edge cases in test data

Example test data file:
```json
{
  "title": "Test Pharmacy",
  "address": "123 Test St, San Francisco, CA 94110",
  "phone": "+14155551234",
  "website": "https://testpharmacy.com"
}
```

## Performance Testing

For performance-critical components, use `pytest-benchmark`:

```python
def test_performance(benchmark):
    result = benchmark(performance_critical_function, test_data)
    assert result == expected_value
```

Run with:
```bash
pytest tests/test_performance.py -v --benchmark-only
```

## Troubleshooting

Common issues and solutions:

1. **Tests hanging**: Check for unclosed resources or infinite loops
2. **Intermittent failures**: Look for race conditions or timing issues
3. **Mocking issues**: Ensure all external calls are properly mocked
4. **Fixture scope**: Check if fixtures need `autouse=True` or different scopes
5. **Test isolation**: Make sure tests don't depend on each other's state