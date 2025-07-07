# Pharmacy Scraper Testing Methodology

This document outlines the comprehensive testing approach used for the Pharmacy Scraper project. Following these practices ensures the pipeline remains stable, reliable, and maintains high-quality code standards.

## Testing Philosophy

The Pharmacy Scraper testing strategy follows these core principles:

1. **Comprehensive Coverage**: Aim for high test coverage across all modules
2. **API Protection**: Never make real API calls in automated tests
3. **Deterministic Results**: Tests should produce consistent results regardless of environment
4. **Isolation**: Each test should be independent of others
5. **Performance**: Tests should execute quickly for rapid development cycles

## Test Directory Structure

```
tests/
├── classification/       # Tests for pharmacy classification components
├── hospital/            # Tests for hospital identification functionality
├── integration/         # End-to-end and cross-component tests
├── orchestrator/        # Tests for pipeline orchestration
└── utils/               # Tests for shared utility functions
```

## Test Types

### 1. Unit Tests

Unit tests focus on testing individual components in isolation. For example:

- Testing the `RateLimiter` class in `perplexity_client.py`
- Testing individual helper functions in `classifier.py`

### 2. Integration Tests

Integration tests verify that multiple components work together correctly:

- Testing the interaction between `Classifier` and `PerplexityClient`
- Verifying cache functionality across pipeline stages

### 3. End-to-End Tests

End-to-end tests validate the entire pipeline workflow:

- `test_end_to_end_pipeline` in `test_orchestrator_advanced.py`
- Tests that simulate a complete pipeline run with mocked external services

## Mocking Strategy

### API Service Mocking

External API calls are never made during testing. Instead, we use these approaches:

#### 1. Direct Method Replacement

For the pipeline orchestrator, we use direct method replacement on the orchestrator instance:

```python
def test_end_to_end_pipeline(orchestrator, test_data):
    # Direct method replacement for external API calls
    orchestrator.classify_pharmacy = mock_classify_pharmacy
    orchestrator.verify_pharmacy = mock_verify_pharmacy
    
    # Run pipeline with mocked methods
    result = orchestrator.run()
    assert result is not None
```

This approach is preferred for orchestrator testing as it ensures stable test behavior.

#### 2. Response Mocking

For API client tests, we mock HTTP responses:

```python
@patch("httpx.AsyncClient.post")
def test_perplexity_api_call(mock_post, client):
    # Configure mock response
    mock_post.return_value.__aenter__.return_value.status_code = 200
    mock_post.return_value.__aenter__.return_value.json.return_value = {
        "id": "test-id",
        "model": "sonar",
        "choices": [{
            "message": {
                "content": "Independent pharmacy"
            }
        }]
    }
    
    # Test the API call
    result = client.make_api_call(pharmacy_data)
    assert "Independent pharmacy" in result
```

#### 3. Fixture-Based Mocking

Test fixtures provide consistent test data:

```python
@pytest.fixture
def mock_pharmacy_data():
    return PharmacyData(
        name="Test Pharmacy",
        address="123 Test St",
        city="Testville",
        state="TS",
        phone="555-123-4567"
    )
```

### Mock Data Storage

Mock data is stored in:

1. **Fixtures**: Defined in `tests/conftest.py` for shared test data
2. **JSON Files**: Located in `tests/test_data/` for complex mock responses
3. **Inline Definitions**: For simpler test cases

## Testing the Perplexity API Client

The Perplexity client has comprehensive test coverage:

### Basic Functionality
- Model initialization and configuration
- Prompt generation
- Response parsing
- Error handling

### Edge Cases
- Rate limiting behavior
- API errors and retries
- Malformed responses
- Network failures

### Testing with Different Models

Tests verify compatibility with different Perplexity models:
- Current `sonar` model
- Legacy models (via parameterized tests)
- Error cases for invalid models

## Testing the Pipeline Orchestrator

The orchestrator tests focus on:

### State Management
- Pipeline initialization
- State tracking across stages
- Resuming from partial completion

### Cache Functionality
- Cache creation and retrieval
- Cache invalidation
- TTL behavior

### Error Handling
- Recovering from stage failures
- Proper error propagation
- Budget enforcement

## Test Helpers and Utilities

### 1. Fixtures

Common test fixtures include:

```python
@pytest.fixture
def orchestrator():
    """Creates a test pipeline orchestrator with mock configuration."""
    config = {
        "locations": [{"city": "TestCity", "state": "TS"}],
        "cache_dir": "./test_cache",
        "output_dir": "./test_output"
    }
    return PipelineOrchestrator(config)
```

### 2. Assertions

Custom assertions help verify complex conditions:

```python
def assert_pharmacy_properly_classified(pharmacy, classification_result):
    """Verifies pharmacy classification meets expected criteria."""
    assert classification_result is not None
    assert hasattr(classification_result, "is_independent")
    assert hasattr(classification_result, "confidence")
    assert 0 <= classification_result.confidence <= 1.0
```

## Running Tests

### Standard Test Run

```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/classification/test_perplexity_client.py

# Run with coverage
pytest --cov=src/pharmacy_scraper tests/
```

### Test Tags

Tests are tagged for selective execution:

- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.slow`: Tests that may take longer to execute

```bash
# Run only unit tests
pytest -m "unit"

# Skip slow tests
pytest -m "not slow"
```

## Continuous Integration

Test automation follows these best practices:

1. **Pre-commit Hooks**: Run linters and formatting checks
2. **CI Pipeline**: Executes full test suite on pull requests
3. **Coverage Reports**: Track test coverage trends over time
4. **Security Scans**: Check for potential vulnerabilities

## Test-Driven Development (TDD)

For new features, follow this TDD workflow:

1. Write failing tests that define the expected behavior
2. Implement the minimal code required to pass the tests
3. Refactor for clarity and performance while maintaining test coverage

## Best Practices

### 1. API Key Safety

- Never hardcode real API keys in tests
- Use placeholders or environment variables for configuration
- Validate that tests can detect when API keys are missing

### 2. Test Isolation

- Clean up resources between tests
- Don't rely on test execution order
- Reset global state before and after each test

### 3. Performance Considerations

- Keep unit tests fast (under 10ms if possible)
- Group slow tests and mark them accordingly
- Use appropriate mocking to avoid network calls

### 4. Documentation

- Include docstrings explaining test purpose
- Document any complex test setups or assertions
- Reference requirements or specifications being tested

## Test Coverage Goals

| Module                 | Target Coverage |
|------------------------|----------------|
| Classification         | ≥ 95%          |
| Orchestrator           | ≥ 90%          |
| Utils                  | ≥ 95%          |
| Overall                | ≥ 90%          |

Current coverage metrics:
- **Classification Module**: 100% for classifier, 92% for Perplexity client
- **Orchestrator Module**: 88% coverage
- **Overall**: 91% coverage

## Future Test Improvements

1. **Property-based Testing**: Implement property tests for edge cases
2. **Fuzzing Tests**: Add fuzz testing for API response parsing
3. **Load Testing**: Simulate high-volume pipeline execution
4. **Snapshot Testing**: Verify complex data structures remain consistent
