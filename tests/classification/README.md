# Classification Module Test Suite

This document provides an overview of the test suite for the Pharmacy Scraper's classification module, explaining the purpose and context of each test file.

## Test Coverage Summary

The classification module test suite achieves approximately 99% test coverage for the core `classifier.py` file and high coverage for related components. Tests cover all major code paths including:

- Rule-based classification (chain/compounding detection)
- LLM-based classification using Perplexity API
- Caching behavior with various input types
- Batch classification functionality
- Error handling and edge cases
- Model override capabilities

## Test Structure

### Core Classification Tests

#### `test_classifier.py`
Primary tests for the `Classifier` class, covering basic classification functionality, rule-based classification, and LLM fallback.

#### `test_classifier_coverage.py`
Comprehensive tests designed to achieve high code coverage, targeting specific code paths and edge cases in `classifier.py`.

#### `test_classify_pharmacies_batch.py`
Tests for the batch classification functionality, verifying behavior with various input types, error handling, and model overrides in batch processing.

### Perplexity Client Tests

#### `test_perplexity_client.py`
Basic tests for the Perplexity API client integration, verifying proper initialization and API calls.

#### `test_perplexity_client_init.py`
Specialized tests for client initialization, covering API key handling, model selection, and configuration options.

#### `test_perplexity_client_response_parsing.py`
Tests for response parsing logic, ensuring proper transformation of LLM responses into `ClassificationResult` objects.

#### `test_perplexity_client_error_handling.py`
Tests for error handling in the Perplexity client, including API errors, rate limiting, and malformed responses.

#### `test_perplexity_client_comprehensive.py`
End-to-end tests for the Perplexity client, covering the full classification workflow.

#### `test_perplexity_client_edge_cases.py`
Tests for edge cases in the Perplexity client, such as unusual pharmacy data and extreme values.

#### `test_make_api_call.py`
Tests focused specifically on the API call functionality, including retries and timeout handling.

#### `test_parse_response.py`
Tests for parsing raw API responses into structured `ClassificationResult` objects.

#### `test_generate_prompt.py`
Tests for prompt generation logic used to create effective prompts for the LLM.

### Cache-related Tests

#### `test_cache.py`
Tests for the classification cache functionality, verifying proper caching behavior and performance benefits.

#### `test_cache_errors.py`
Tests for error handling in the cache system, including corrupt cache entries and serialization issues.

#### `test_cache_limits.py`
Tests for cache size limits and eviction policies.

#### `test_rate_limiter.py`
Tests for API rate limiting functionality to prevent exceeding API quotas.

### Data Model Tests

#### `test_models.py`
Tests for the data models used throughout the classification system, including `PharmacyData` and `ClassificationResult`.

## Test Fixtures

The test suite uses several key fixtures:

- `mock_perplexity_client`: A mock implementation of the PerplexityClient for testing without making real API calls
- `mock_pharmacies`: Sample pharmacy data for testing classification logic
- `classifier`: A properly initialized Classifier instance for integration testing

## Common Test Patterns

1. **Mocking Strategy**:
   - External API calls are always mocked to prevent actual API usage during tests
   - Cache is often cleared or bypassed to ensure consistent test results

2. **Coverage Approach**:
   - Unit tests for individual components
   - Integration tests for component interactions
   - Edge case testing for error handling

3. **Assertion Patterns**:
   - Verification of classification result properties
   - Validation of cache behavior
   - Confirmation of proper API call patterns
   - Checking of error handling and fallback behavior

## Known Issues and Special Cases

1. **Parameter Mismatch**: There is a mismatch between `rule_based_classify` creating `ClassificationResult` with a `method` parameter that doesn't exist in the class definition. Tests work around this issue by using mocks.

2. **Cache Interference**: Tests that rely on specific cache behavior clear the cache before execution to prevent interference between tests.

3. **Token Match Function**: The `_token_match` function is defined as an inner function inside `rule_based_classify`, making direct testing challenging. Tests work around this by testing the behavior rather than the function directly.

## Running the Tests

```bash
# Run all classification tests
pytest tests/classification/

# Run specific test file
pytest tests/classification/test_classifier.py

# Run with coverage
pytest --cov=src/pharmacy_scraper/classification --cov-report=term-missing tests/classification/

# Run specific test case
pytest tests/classification/test_classifier.py::TestClassifier::test_classify_pharmacy
```
