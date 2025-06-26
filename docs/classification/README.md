# Pharmacy Scraper: Classification Module  

## Overview  
The classification module handles the categorization of pharmacies using the Perplexity API. It provides a robust client implementation with features like request retries, rate limiting, and response caching.  

## Project Status  
- **Current Focus**: Perplexity client testing and validation  
- **Module**: Classification system (`perplexity_client.py`)  
- **Branch**: `feature/independent-pharmacy-filter`  
- **Test Status**: All `_make_api_call` tests passing  

## Features  
- 🚀 High-performance API client for Perplexity AI  
- 🔄 Automatic retry on rate limits  
- 💾 Response caching to reduce API calls  
- 🧪 Comprehensive test coverage  
- 🧵 Thread-safe implementation  

## Recent Improvements  
- Added comprehensive test suite for `PerplexityClient._make_api_call`  
- Fixed test assertions to match actual implementation behavior  
- Implemented proper mocking for OpenAI client and exceptions  
- Added tests for success cases, retries, and error handling  
- Verified correct API key usage and request headers  

## Getting Started  

### Prerequisites  
- Python 3.9.6+  
- Perplexity API key  

### Installation  
```bash
# Install dependencies
pip install -r requirements.txt
```

### Configuration  
Set your Perplexity API key in your environment:  
```bash
export PERPLEXITY_API_KEY='your-api-key-here'
```

## Usage  

### Basic Example  
```python
from pharmacy_scraper.classification.perplexity_client import PerplexityClient

# Initialize client
client = PerplexityClient(api_key="your-api-key")

# Make an API call
response = client._make_api_call(
    pharmacy_data={"name": "Example Pharmacy"},
    model="sonar",
    max_tokens=1000
)
```

### Testing  
Run the full test suite:  
```bash
# Run all classification tests  
pytest tests/classification/ -v  

# Run specific test file  
pytest tests/classification/test_make_api_call.py -v  

# Run with coverage report  
pytest --cov=src/pharmacy_scraper/classification tests/classification/  
```

## Architecture  

### Key Components  
- `PerplexityClient`: Main client class handling API interactions  
- `RateLimiter`: Manages rate limiting and retries  
- `ResponseCache`: Handles response caching  
- `Data Models`: Pydantic models for request/response validation  

### Error Handling  
The client implements comprehensive error handling for:  
- Rate limiting (automatic retries)  
- Invalid responses  
- Network errors  
- Authentication failures  

## Testing Approach  
- **Unit Tests**: Isolated testing of individual components  
- **Mocks**: OpenAI client and exceptions  
- **Fixtures**: Reusable test data and client setup  
- **Assertions**: Verify both success and error paths  

## Next Steps  
1. Add tests for cache read/write functionality  
2. Implement tests for batch processing logic  
3. Add integration tests for full classification workflow  
4. Document the testing approach and patterns  

## Key Files  
- `src/pharmacy_scraper/classification/perplexity_client.py`  
- `tests/classification/test_make_api_call.py`  
- `tests/classification/test_parse_response.py`  
- `tests/classification/test_rate_limiter.py`  

## Dependencies  
- Python 3.9.6+  
- `pytest` (for testing)  
- `pytest-cov` (coverage reporting)  
- `unittest.mock` (test mocking)  
- `pydantic` (data validation)  

## Contributing  
1. Fork the repository  
2. Create a feature branch  
3. Commit your changes  
4. Push to the branch  
5. Create a new Pull Request  

## License  
[Specify License]  
