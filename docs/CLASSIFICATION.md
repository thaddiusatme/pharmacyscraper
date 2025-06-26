# Classification Module

This module provides functionality for classifying pharmacies using a combination of rule-based and LLM-based approaches. It includes a caching system to improve performance and reduce API calls.

## Table of Contents
- [Overview](#overview)
- [Cache System](#cache-system)
- [Usage](#usage)
- [Configuration](#configuration)
- [Error Handling](#error-handling)
- [Performance Considerations](#performance-considerations)

## Overview

The classification module uses the Perplexity API to classify pharmacies based on their details. It includes:

- **PerplexityClient**: Main client for interacting with the Perplexity API
- **Caching**: File-based caching of classification results
- **Rate Limiting**: Built-in rate limiting to respect API quotas
- **Retry Logic**: Automatic retries for failed requests

## Cache System

The cache system stores classification results to avoid redundant API calls. Each cache entry is stored as a JSON file with a filename generated from a hash of the pharmacy data and model name.

### Cache Location
- Default: `data/cache/classification/`
- Custom: Specify any directory path when initializing the client

### Cache Invalidation
- **Automatic**: Cache entries never expire (files must be manually deleted or the cache directory cleared)
- **Force Refresh**: Set `force_reclassification=True` to bypass the cache

### Cache Key Generation
Cache keys are generated using SHA-256 hashing of:
- Pharmacy name/title
- Pharmacy address
- Model name

## Usage

### Basic Usage

```python
from pharmacy_scraper.classification import PerplexityClient

# Initialize with default settings
client = PerplexityClient(api_key="your_api_key")

# Classify a pharmacy
result = client.classify_pharmacy({
    "title": "Test Pharmacy",
    "address": "123 Test St, Test City, TS 12345",
    "phone": "(555) 123-4567"
})
```

### Cache Configuration

```python
# Custom cache directory
client = PerplexityClient(
    api_key="your_api_key",
    cache_dir="/path/to/custom/cache"
)

# Disable caching
client = PerplexityClient(api_key="your_api_key", cache_dir=None)

# Force reclassification (bypass cache)
client = PerplexityClient(
    api_key="your_api_key",
    force_reclassification=True
)
```

## Configuration

### PerplexityClient Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | str | None | Perplexity API key (required) |
| `model_name` | str | "sonar" | Model to use for classification |
| `rate_limit` | int | 20 | Max requests per minute |
| `cache_dir` | str/None | "data/cache/classification" | Cache directory, or None to disable |
| `force_reclassification` | bool | False | Bypass cache and force new classification |
| `max_retries` | int | 3 | Max retry attempts for failed requests |

## Error Handling

The module handles various error conditions:

- **Rate Limiting**: Automatically retries with exponential backoff
- **API Errors**: Logs detailed error information
- **Cache Errors**: Logs warnings but continues operation
- **Network Issues**: Implements retry logic with backoff

## Performance Considerations

- **API Calls**: Each unique classification requires an API call (unless cached)
- **Cache Performance**: Cache hits are typically <10ms
- **Rate Limiting**: Default limit is 20 requests per minute
- **Thread Safety**: The client is thread-safe for concurrent requests

## Testing

Run the classification tests with:

```bash
# Run all classification tests
pytest tests/classification/ -v

# Run cache-specific tests
pytest tests/classification/test_cache.py -v
```
