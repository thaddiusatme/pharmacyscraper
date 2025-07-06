# Pharmacy Scraper: Orchestrator Module

## Project Status
- **Current Focus**: Orchestrator test coverage improvements and documentation
- **Module**: Pipeline Orchestrator system (pipeline_orchestrator.py)
- **Branch**: main
- **Test Status**: All tests passing (88% coverage for orchestrator module)

## Recent Improvements
- Added comprehensive tests for cache functionality
- Added test coverage for cache miss, invalid cache format, and directory creation
- Created detailed documentation for the orchestrator module
- Enhanced coverage from 31% to 88% for the orchestrator
- Updated project README with test coverage information

## Current Context
- Working in: [tests/orchestrator/test_cache_functionality.py](/Users/thaddius/repos/pharmacyscraper/tests/orchestrator/test_cache_functionality.py)
- Test coverage: 88% (pipeline_orchestrator.py)
- Documentation: Newly created with examples and detailed behavior

## Next Steps
1. Add tests for error handling in pipeline stages
2. Implement tests for API budget limit scenarios
3. Add integration tests for full pipeline resumability
4. Consider tests for edge cases in state management

## Key Files
- [src/pharmacy_scraper/orchestrator/pipeline_orchestrator.py](/Users/thaddius/repos/pharmacyscraper/src/pharmacy_scraper/orchestrator/pipeline_orchestrator.py)
- [tests/orchestrator/test_cache_functionality.py](/Users/thaddius/repos/pharmacyscraper/tests/orchestrator/test_cache_functionality.py)
- [tests/orchestrator/test_orchestrator_integration.py](/Users/thaddius/repos/pharmacyscraper/tests/orchestrator/test_orchestrator_integration.py)
- [src/pharmacy_scraper/orchestrator/README.md](/Users/thaddius/repos/pharmacyscraper/src/pharmacy_scraper/orchestrator/README.md)

## Quick Start
```bash
# Run all orchestrator tests
pytest tests/orchestrator/ -v

# Check coverage
pytest --cov=src/pharmacy_scraper/orchestrator --cov-report=term-missing tests/orchestrator/

# Run specific cache functionality tests
pytest tests/orchestrator/test_cache_functionality.py -v
```

## Dependencies
- Python 3.9.6+
- Apify API (for data collection)
- Google Places API (for verification)
- Perplexity API (for LLM classification)
- Pytest (testing)
- pytest-cov (coverage reporting)

## Recent Changes
- Added test_cache_miss to verify orchestrator creates cache on first run
- Added test_invalid_cache_format to verify handling of corrupted cache
- Added test_cache_directory_creation to test auto-creation of directories
- Created comprehensive documentation for the orchestrator module
- Updated main project README with test coverage metrics
