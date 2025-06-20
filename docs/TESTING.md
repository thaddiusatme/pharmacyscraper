# Testing Guide

This document provides detailed information about the test suite and how to run tests for the Independent Pharmacy Verification project.

## Table of Contents
- [Test Structure](#test-structure)
- [Running Tests](#running-tests)
- [Test Coverage](#test-coverage)
- [Writing Tests](#writing-tests)
- [Mocking External Services](#mocking-external-services)
- [Continuous Integration](#continuous-integration)

## Test Structure

The test suite is organized as follows:

```
tests/
├── __init__.py         # Makes test directory a Python package
├── conftest.py         # Pytest fixtures and configuration
└── test_apify_collector.py  # Tests for the Apify collector
```

## Running Tests

### Basic Test Execution

Run all tests:
```bash
pytest tests/
```

Run tests with verbose output:
```bash
pytest -v tests/
```

Run a specific test file:
```bash
pytest tests/test_apify_collector.py
```

Run a specific test function:
```bash
pytest tests/test_apify_collector.py::test_run_collection_success
```

### Test Coverage

Generate a coverage report:
```bash
pytest --cov=scripts tests/
```

Generate an HTML coverage report:
```bash
pytest --cov=scripts --cov-report=html tests/
open htmlcov/index.html  # View the report
```

## Writing Tests

### Test Naming Conventions
- Test files should be named `test_*.py` or `*_test.py`
- Test functions should be prefixed with `test_`
- Test classes should be prefixed with `Test`

### Example Test

```python
def test_example():
    """Test that 1 + 1 equals 2."""
    assert 1 + 1 == 2
```

### Fixtures

Pytest fixtures are defined in `conftest.py` and can be used across test files.

Common fixtures:
- `apify_collector`: Pre-configured ApifyCollector instance
- `mock_apify_client`: Mock Apify client
- `sample_states_cities`: Sample test data

## Mocking External Services

### Apify Client Mocking

The test suite includes a mock Apify client that simulates the Apify API. This allows tests to run without making actual API calls.

Example of using the mock client:

```python
def test_with_mock(mock_apify_client):
    # The mock client is automatically injected
    assert mock_apify_client is not None
```

## Continuous Integration

The project includes a GitHub Actions workflow (`.github/workflows/tests.yml`) that runs the test suite on every push and pull request.

### Running CI Locally

You can run the CI pipeline locally using [act](https://github.com/nektos/act):

```bash
act -P ubuntu-latest=nektos/act-environments-ubuntu:18.04
```

## Debugging Tests

To debug a failing test:

1. Run the test with `-s` to disable output capturing:
   ```bash
   pytest -s tests/test_file.py::test_function
   ```

2. Add print statements or use `pdb`:
   ```python
   import pdb; pdb.set_trace()  # Add this where you want to start debugging
   ```

3. Use `pytest --pdb` to drop into the debugger on failure:
   ```bash
   pytest --pdb tests/test_file.py
   ```