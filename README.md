# Pharmacy Scraper

A Python-based tool for scraping and analyzing pharmacy data from various sources, with a focus on identifying independent, non-hospital pharmacies.

> **Latest Update (v2.0.0)**: Enhanced classification system with improved accuracy for identifying independent pharmacies. Now using Perplexity's `sonar` model for better results.

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

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/pharmacy-scraper.git
   cd pharmacy-scraper
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -e .
   ```

## Configuration

1. Copy the example config file and update with your API keys:
   ```bash
   cp config/example_config.json config/config.json
   ```

2. Update the following in `config/config.json`:
   - Apify API token
   - Google Places API key
   - Other configuration parameters as needed

## Usage

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
pytest tests/ -v
```

### Code Style and Linting

```bash
# Run black formatting
black src/ tests/

# Run flake8 linting
flake8 src/ tests/

# Run type checking
mypy src/
```

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

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
