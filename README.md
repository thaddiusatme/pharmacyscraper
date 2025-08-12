# Pharmacy Scraper

A Python-based tool for scraping and analyzing pharmacy data from various sources, with a focus on identifying independent, non-hospital pharmacies.

> **Latest Update (v2.1.0)**: Production-ready pipeline with secure API key handling, automated testing, and Perplexity `sonar` model integration. Now includes robust budget enforcement and extensive test coverage for the orchestrator (88%).

## Features

### Pipeline Phases

The system processes data through the following phases in order:

1. **Data Collection** (Phase 1)
   - Scrapes pharmacy data using Apify and Google Places API
   - Handles rate limiting and API quotas
   - Implements caching to minimize redundant requests

2. **Deduplication** (Phase 1.5)
   - Smart duplicate removal using fuzzy matching
   - Self-healing capabilities for data gaps
   - Maintains data integrity across sources

3. **Classification** (Phase 2a)
   - AI-powered classification using Perplexity's `sonar` LLM model
   - Enhanced identification of independent vs. chain pharmacies with improved accuracy
   - Supports both rule-based and LLM-based classification with automatic fallback
   - Implements intelligent caching for cost efficiency

4. **Verification** (Phase 2b - Optional)
   - Address and business verification using Google Places
   - Ensures data accuracy and completeness
   - Validates pharmacy details against trusted sources

### Core Capabilities

- **Budget Management**: Tracks and manages API usage and costs
- **Error Handling**: Robust error recovery and retry mechanisms
- **Performance**: Optimized for large-scale data processing
- **Extensibility**: Modular design for easy integration of new data sources

## Module Documentation

Detailed documentation is available for each module:

- [Orchestrator Module](src/pharmacy_scraper/orchestrator/README.md) - Pipeline coordination and cache management
- [Classification Module](tests/classification/README.md) - AI-based pharmacy classification

## Installation

1. Clone the repository:
  ```bash
  git clone https://github.com/your-username/pharmacy-scraper.git
  cd pharmacy-scraper
  ```

2. Create and activate a local virtual environment (recommended):
  ```bash
  python -m venv .venv
  source .venv/bin/activate  # Windows: .venv\Scripts\activate
  ```

3. Install pinned runtime and dev/test dependencies:
  ```bash
  python -m pip install -U pip
  python -m pip install -r requirements.txt -r requirements-dev.txt
  ```

4. Verify tests run locally:
  ```bash
  PYTHONPATH=src python -m pytest -q
  ```

## Configuration

### Option 1: Environment Variables (Recommended for Production)

1. Create a `.env` file in the project root:
   ```bash
   # Create .env file (this file should never be committed to version control)
   touch .env
   ```

2. Add your API keys to the `.env` file:
   ```
   # API Keys
   GOOGLE_MAPS_API_KEY=your_google_api_key_here
   APIFY_API_TOKEN=your_apify_token_here
   PERPLEXITY_API_KEY=your_perplexity_api_key_here
   ```

3. Use the secure production configuration:
  ```bash
  # Make the setup script executable
  chmod +x scripts/setup_env_and_run.sh

  # Run the pipeline securely
  ./scripts/setup_env_and_run.sh
  ```

### Option 2: Configuration File

1. Copy the example config file and update with your API keys:
   ```bash
   cp config/example_config.json config/config.json
   ```

2. Update the following in `config/config.json`:
   - Apify API token
   - Google Places API key
   - Perplexity API key
   - Other configuration parameters as needed

## Usage

### Running the Production Pipeline

The pharmacy scraper now supports two main execution modes:

1. **Test Pipeline** (No API Calls):
  ```bash
  python scripts/run_test_pipeline.py
  ```
   This runs the pipeline with mocked API services, perfect for testing changes without using API credits.

2. **Production Pipeline** (Real API Calls):
  ```bash
  # Run with default configuration
  python scripts/run_production_pipeline.py
  
  # Reset state and start fresh
  python scripts/run_production_pipeline.py --reset
  
  # Use a specific configuration
  python scripts/run_production_pipeline.py --config config/production/custom_config.json
  
  # Validate configuration without making API calls
  python scripts/run_production_pipeline.py --dry-run
  ```

3. **Secure Environment Setup** (Recommended):
  ```bash
  # This uses environment variables from .env file
  ./scripts/setup_env_and_run.sh
  ```

### API Budget Management

The pipeline enforces API budget limits specified in your configuration:

```json
"max_budget": 50.0,
"api_cost_limits": {
  "apify": 0.5,
  "google_places": 0.3,
  "perplexity": 0.2
}
```

This prevents unexpected API costs while ensuring data quality.

### Programmatic Usage

The primary entry point for classifying pharmacies is the `Classifier` class. It provides a simple interface that handles rule-based classification, LLM fallback, and caching automatically.

### Basic Classification

Here's how to classify a single pharmacy:

```python
from pharmacy_scraper.classification import Classifier
from pharmacy_scraper.classification.data_models import PharmacyData

# 1. Instantiate the classifier
# The PerplexityClient is created and managed internally.
classifier = Classifier()

# 2. Define the pharmacy data
pharmacy_data = {
    "name": "Downtown Pharmacy",
    "address": "123 Main St",
    "city": "San Francisco",
    "state": "CA",
    "zip": "94105",
    "phone": "(555) 123-4567"
}

# 3. Classify the pharmacy
# The `use_llm` flag is True by default, enabling LLM fallback for low-confidence results.
result = classifier.classify_pharmacy(pharmacy_data, use_llm=True)

# 4. Inspect the result
print(f"Is Chain: {result.is_chain}")
print(f"Confidence: {result.confidence}")
print(f"Reason: {result.reason}")
print(f"Method: {result.method.value}") # 'rule-based', 'llm', or 'cached'
```

### Batch Classification

To classify multiple pharmacies, simply iterate and call `classify_pharmacy`. The internal cache will prevent redundant API calls for duplicate entries.

```python
pharmacies = [
    {"name": "Downtown Pharmacy", "address": "123 Main St"},
    {"name": "CVS Pharmacy #1234", "address": "456 Market St"},
    # ... more pharmacies
]

results = [classifier.classify_pharmacy(p) for p in pharmacies]

for r in results:
    print(f"{r.reason} (Method: {r.method.value})")
```

### Caching

Caching is enabled by default and managed internally by the `Classifier`. Results are stored in a local cache to minimize API costs and improve performance. There is no need for manual cache configuration.
client = PerplexityClient(
    api_key="your_api_key",
    force_reclassification=True
)
```

## Usage

### Running the Pipeline

```bash
python -m pharmacy_scraper.run_pipeline --config config/your_config.json
```

### Running Tests

```bash
PYTHONPATH=src pytest tests/ -v
```

### Testing Quick Start

Common testing commands (see detailed guide in [`docs/TESTING.md`](docs/TESTING.md)):

- Full suite:
  ```bash
  make test
  ```
- QA suites (integration/contract/property):
  ```bash
  make test-qa
  ```
- Performance benchmarks (opt-in):
  ```bash
  make test-perf           # measure only (PERF=1)
  make test-perf-strict    # enforce thresholds (PERF=1 PERF_STRICT=1)
  ```

### Test Coverage

The project currently has 73% overall test coverage, with key modules having excellent coverage:

- **Orchestrator Module**: 88% coverage
  - Comprehensive cache functionality tests
  - Pipeline state management tests
  - Stage execution tests
- **Classification Module**: 100% coverage for classifier, 92% for Perplexity client
- **State Manager**: 96% coverage

## Project Structure

```
pharmacy-scraper/
├── config/                   # Configuration files
├── data/                     # Data files (ignored in git)
├── docs/                     # Documentation
├── scripts/                  # Scripts for data collection and processing
├── src/                      # Source code
│   ├── pharmacy_scraper/     # Main package
│   │   ├── api/              # API clients
│   │   ├── classification/   # AI classification
│   │   ├── config/           # Configuration
│   │   └── ...
├── tests/                   # Test files
├── .github/                 # GitHub workflows and templates
├── .gitignore
├── pyproject.toml
└── README.md
```

## Context Engineering (Windsurf)

This repository is configured for AI-assisted development with Windsurf:

- The `.windsurf/` directory contains rules and archives to help the assistant prioritize relevant context.
- `pyproject.toml` sets `pytest` options with `pythonpath = ["src"]`, so modules import correctly during tests.
- Use a local virtual environment (`.venv`) and the pinned `requirements*.txt` to ensure deterministic behavior.
- Common commands:
  ```bash
  # Activate env
  source .venv/bin/activate

  # Run full test suite
  PYTHONPATH=src python -m pytest -q

  # Lint/format (optional)
  black . && flake8
  ```

If you encounter plugin-related pytest issues, you can temporarily run with:
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest -q
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Dependency and Security Checks

For consistent environments, runtime dependencies are pinned in `requirements.txt`, and development/test tools are pinned in `requirements-dev.txt`.

Run basic security and dependency audits locally:

```bash
pip install -r requirements.txt -r requirements-dev.txt
bash scripts/dev_security_check.sh
```

This script runs `pip-audit` for known vulnerabilities and `detect-secrets` to scan for accidentally committed secrets.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
