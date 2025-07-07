# Pharmacy Scraper Production Pipeline Guide

This document outlines how to use and configure the pharmacy scraper production pipeline for real-world data collection and classification.

## Overview

The production pipeline enables end-to-end pharmacy data collection, deduplication, classification, and verification using real API calls. It includes:

- Secure API key handling via environment variables
- Budget enforcement to prevent unexpected costs
- Comprehensive logging and error handling
- Dry-run capability for configuration validation
- State management for resumable operations

## Prerequisites

Before running the production pipeline, ensure you have:

1. API credentials for:
   - Apify API (`APIFY_TOKEN` or `APIFY_API_TOKEN`)
   - Google Maps API (`GOOGLE_MAPS_API_KEY`)
   - Perplexity API (`PERPLEXITY_API_KEY`)

2. Python environment with required dependencies:
   ```bash
   pip install -e .
   ```

3. Properly configured `.env` file (recommended) or configuration JSON.

## Running Options

### Option 1: Using Environment Variables (Recommended)

Create a `.env` file in the project root:

```
# API Keys
GOOGLE_MAPS_API_KEY=your_google_api_key_here
APIFY_API_TOKEN=your_apify_token_here
PERPLEXITY_API_KEY=your_perplexity_api_key_here
```

Then run the setup script:

```bash
./setup_env_and_run.sh
```

This script will:
1. Load API keys from the `.env` file
2. Validate required environment variables
3. Create necessary directories
4. Execute the production pipeline

### Option 2: Using `run_production_pipeline.py`

Run the pipeline directly with various options:

```bash
# Basic run (uses default configuration)
python run_production_pipeline.py

# Reset pipeline state and start fresh
python run_production_pipeline.py --reset

# Use a specific configuration file
python run_production_pipeline.py --config config/production/custom_config.json

# Test configuration without making API calls
python run_production_pipeline.py --dry-run
```

## Configuration Files

The production pipeline uses configuration files to determine:

- Locations to search
- API budget limits
- Cache settings
- Output locations

Example production configuration (`config/production/secure_production_config.json`):

```json
{
  "locations": [
    {"city": "Phoenix", "state": "AZ"},
    {"city": "Tucson", "state": "AZ"},
    {"city": "Los Angeles", "state": "CA"}
  ],
  "queries": ["pharmacy", "drugstore", "apothecary"],
  "max_budget": 50.0,
  "api_cost_limits": {
    "apify": 0.5,
    "google_places": 0.3,
    "perplexity": 0.2
  },
  "api_keys": {
    "apify_token": "${APIFY_TOKEN}",
    "google_places_key": "${GOOGLE_MAPS_API_KEY}",
    "perplexity_key": "${PERPLEXITY_API_KEY}"
  },
  "cache_dir": "./cache/production",
  "output_dir": "./output/production",
  "log_level": "INFO"
}
```

Note the use of environment variable placeholders (`${VAR_NAME}`), which are automatically replaced with actual values from the environment.

## Budget Management

The pipeline strictly enforces budget limits to prevent unexpected API costs:

- `max_budget`: Maximum total cost allowed for the entire pipeline run
- `api_cost_limits`: Cost per operation for each API service

When a budget limit is reached, the pipeline will:
1. Log a warning message
2. Skip the affected operation
3. Continue processing where possible
4. Save all processed data up to that point

## Testing Without API Costs

For development and testing without incurring API costs, use the test pipeline:

```bash
python run_test_pipeline.py
```

This runs with mocked API services and a small sample configuration.

## Troubleshooting

### Common Issues

1. **Missing API Keys**:
   - Ensure all required environment variables are set
   - Check for correct variable names (`APIFY_TOKEN` or `APIFY_API_TOKEN`)

2. **Budget Limits Reached**:
   - Increase budget limits in configuration file
   - Run with `--reset` to reset the API credit tracker

3. **Cache Files Missing**:
   - Run with `--reset` to start fresh
   - Verify cache directory permissions

4. **API Errors**:
   - Check API key validity
   - Verify API service status
   - Review logs for specific error messages

### Log File

The pipeline generates detailed logs that can help diagnose issues:

```bash
# View the last 100 lines of logs
tail -n 100 ./logs/pharmacy_scraper.log
```

## Performance Considerations

For optimal performance and cost efficiency:

1. Start with smaller geographic areas to validate pipeline behavior
2. Use the `--dry-run` option to validate configuration before actual execution
3. Configure appropriate cache TTL settings for your update frequency
4. Monitor API credit usage to optimize budget allocation
