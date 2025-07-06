# Pharmacy Scraper: Classification Module

## Project Status
- **Current Focus**: Orchestrator module test coverage and documentation
- **Module**: Classification system (perplexity_client.py)
- **Branch**: main
- **Test Status**: All tests passing (92% coverage for perplexity_client.py, 100% for classifier.py)

## Recent Improvements
- Fixed failing tests in orchestrator test suite using direct method replacement
- Added comprehensive cache functionality tests to the orchestrator module
- Improved test coverage for cache miss, invalid formats, and directory creation
- Created detailed documentation for both orchestrator and classification modules
- Enhanced overall project test coverage to 73%

## Current Context
- Working in: [tests/orchestrator/test_cache_functionality.py](/Users/thaddius/repos/pharmacyscraper/tests/orchestrator/test_cache_functionality.py)
- Test coverage: 92% (perplexity_client.py), 100% (classifier.py)
- Documentation: Updated project README and added module-specific documentation

## Next Steps
1. Review and improve test coverage for remaining untested code paths
2. Add integration tests for the full classification workflow
3. Implement tests for API budget limit scenarios
4. Add tests for error handling in pipeline stages

## Key Files
- [src/pharmacy_scraper/classification/perplexity_client.py](/Users/thaddius/repos/pharmacyscraper/src/pharmacy_scraper/classification/perplexity_client.py)
- [src/pharmacy_scraper/orchestrator/pipeline_orchestrator.py](/Users/thaddius/repos/pharmacyscraper/src/pharmacy_scraper/orchestrator/pipeline_orchestrator.py)
- [tests/orchestrator/test_cache_functionality.py](/Users/thaddius/repos/pharmacyscraper/tests/orchestrator/test_cache_functionality.py)
- [tests/classification/test_perplexity_client_comprehensive.py](/Users/thaddius/repos/pharmacyscraper/tests/classification/test_perplexity_client_comprehensive.py)

## Quick Start
```bash
# Run all classification tests
pytest tests/classification/test_perplexity_*.py -v

# Run all orchestrator tests
pytest tests/orchestrator/ -v

# Check classification coverage
pytest --cov=src/pharmacy_scraper/classification --cov-report=term-missing tests/classification/

# Check orchestrator coverage
pytest --cov=src/pharmacy_scraper/orchestrator --cov-report=term-missing tests/orchestrator/
```

## Dependencies
- Python 3.9.6+
- Perplexity API (for LLM classification)
- Apify API (for data collection)
- Google Places API (for verification)
- OpenAI Python client (Perplexity API wrapper)
- Pydantic (data validation)
- Pytest (testing)
- pytest-cov (coverage reporting)

## Recent Changes
- Added comprehensive test coverage for cache functionality in the orchestrator
- Created documentation for the orchestrator module with examples
- Updated project README with test coverage metrics
- Fixed test issues using direct method replacement instead of function patching
- Improved validation for cache directory creation and invalid cache handling
