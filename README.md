# 🏥 Pharmacy Scraper

A comprehensive tool for collecting, processing, and verifying independent pharmacy information at scale. This project includes AI-powered classification, data deduplication, and automated data collection capabilities.

## 🚀 Features

- **AI-Powered Classification**: Uses Perplexity API with local caching for efficient classification of pharmacies
- **Data Collection**: Automated data collection using Apify and Google Maps APIs
- **Deduplication**: Smart deduplication to ensure data quality
- **Self-Healing Pipeline**: Automatically fills gaps in under-populated states
- **Scalable Architecture**: Designed to handle large-scale data collection across all 50 US states

## 🏗️ Project Structure

```
pharmacy_scraper/
├── config/                 # Configuration files
│   ├── production/        # Production configurations
│   └── development/       # Development configurations
├── data/                   # Data files (gitignored)
│   ├── raw/               # Raw data from sources
│   ├── processed/         # Processed data
│   └── cache/             # Cached data (API responses, etc.)
├── docs/                  # Documentation
├── notebooks/             # Jupyter notebooks for analysis
├── scripts/               # Utility scripts (one-off/analysis)
│   ├── analysis/         # Data analysis scripts
│   ├── data_processing/   # Data processing utilities
│   └── utils/            # General utilities
├── src/                   # Source code (core functionality)
│   ├── api/              # API clients and integrations
│   ├── classification/    # AI/rule-based classification
│   ├── data_processing/   # Data transformation
│   ├── models/           # Data models
│   └── utils/            # Utility functions
└── tests/                 # Tests
    ├── unit/             # Unit tests
    └── integration/      # Integration tests
```

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd pharmacy-scraper
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

## 🚦 Usage

### Running the Pipeline

```bash
# Run the full pipeline
python -m src.run_pipeline --config config/production/pipeline_config.json

# Run classification only
python -m src.classification.classifier --input data/input/pharmacies.csv --output data/output/classified.csv
```

### Utility Scripts

```bash
# Analyze trial results
python -m scripts.analysis.analyze_trial --input data/trial_results/

# Process cached data
python -m scripts.data_processing.process_cached_data --states ca,ny,tx
```

## 📊 Example Results

### Classification Performance
- **CA/WA/OR batch:** 230 pharmacies, 123 API calls, 77 chains, 27 hospitals pre-filtered
- **NY/TX/FL/IL/PA batch:** 387 pharmacies, 338 API calls, 60 chains pre-filtered

### Key Metrics
- **Cost savings:** 40-60% reduction in API calls through smart pre-filtering
- **Accuracy:** >95% classification accuracy on test sets
- **Scalability:** Successfully tested on batches of 2,500+ pharmacies

## 🤖 How It Works

1. **Data Collection**:
   - Uses Apify to collect pharmacy data from various sources
   - Supports Google Maps API for additional verification

2. **Pre-processing**:
   - Deduplicates records
   - Standardizes addresses and phone numbers
   - Pre-filters known chains and hospitals

3. **Classification**:
   - Uses Perplexity API for AI-powered classification
   - Implements local caching to reduce API calls
   - Applies post-processing rules for final verification

4. **Output**:
   - Generates clean, classified datasets
   - Produces summary statistics and reports
   - Supports multiple output formats (CSV, JSON, Excel)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📬 Contact

For questions or support, please open an issue on the repository.

### How to Run

To execute a test run, use the following command:

```bash
caffeinate -i python3 -m scripts.run_pipeline \
    --config config/five_state_run.json \
    --output data/five_state_results \
    --log-file data/five_state_results/pipeline.log \
    --skip_verify
```



### Classification System & Verification
- **Perplexity Client**: Enhanced with 83% test coverage and robust error handling
- **Google Places Integration**: Added address verification with 90% success rate
- **Test Suite**: Comprehensive edge case testing for classification
- **Phoenix Trial**: Successful end-to-end pipeline test with Phoenix, AZ data

### Pipeline Improvements
- **End-to-End Testing**: Verified complete workflow from data collection to verification
- **Error Handling**: Added retry logic and improved error recovery
- **Documentation**: Updated project docs and test coverage reports

### Trial Run Fixes & Optimization 
- **Fixed Apify Actor Input Validation**: Resolved `allPlacesNoSearchAction` field error (changed from "false" to empty string)
- **Credit Usage Optimization**: Added `forceExit: true` and `maxCrawledPlacesPerSearch` limits to control costs
- **Successful Trial Results**: Collected 39 independent pharmacies across CA (19) and TX (20) in 4 major cities
- **Configuration Updates**: Optimized `trial_config.json` for targeted data collection (~25 locations per state)
- **Performance**: All actor calls now succeed (eliminated 400 Bad Request errors)

### Apify Collector Improvements
- **Enhanced Test Coverage**: Added comprehensive tests for the Apify collector with proper mocking
- **Improved Error Handling**: Better error messages and recovery for API failures
- **Config Flexibility**: Support for both simple query strings and structured location-based queries
- **Testing Improvements**:
  - Fixed mock data structures to match real API responses
  - Added proper test cleanup and isolation
  - Improved test assertions and coverage

## Project Overview

This project collects and verifies information about independent pharmacies across the United States using a scalable, end-to-end data pipeline.

## Usage

To execute the full data pipeline, use the `scripts/run_pipeline.py` script. It is recommended to start with a smaller, controlled test run before launching a large-scale collection.

**Example: Running a 5-State Test**

This command will run the pipeline for 5 states, prevent the computer from sleeping, save results and logs to `data/five_state_results`, and skip the costly verification step.

```bash
caffeinate -i python3 -m scripts.run_pipeline \
    --config config/five_state_run.json \
    --output data/five_state_results \
    --log-file data/five_state_results/pipeline.log \
    --skip_verify
```

## Features
- Automated data collection from Google Maps via Apify
- Comprehensive test suite with proper mocking and isolation
- Support for both simple queries and structured location-based searches
- Data validation and cleaning
- Chain pharmacy filtering
- CSV export functionality



## 🛠 Setup

1. Clone the repository
   ```bash
   git clone <repository-url>
   cd pharmacy-verification
   ```

2. Set up the environment
   ```bash
   # Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. Configure environment variables
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit .env and add your Apify API token
   APIFY_TOKEN=your_apify_token_here
   ```

## Running Tests

Run the full test suite:
```bash
pytest tests/ -v
```

Run tests with coverage report:
```bash
pytest --cov=src --cov=scripts tests/
```

## Project Structure

```
.
├── config/             # Configuration files for different run scales
│   ├── trial_config.json
│   ├── five_state_run.json
│   └── large_scale_run.json
├── data/               # Output directory for pipeline results
├── scripts/            # Main scripts for execution
│   ├── run_pipeline.py     # End-to-end pipeline orchestration
│   └── apify_collector.py  # Core Apify data collection logic
├── src/                # Source code for core logic
│   ├── classification/
│   ├── dedup_self_heal/
│   ├── utils/
│   └── verification/
├── tests/              # Test suite
├── .api_cache/         # Caching for API calls (e.g., classification)
├── .env.example        # Example environment variables
├── requirements.txt    # Python dependencies
└── README.md           # This file
```



## Development

### Adding New Features
1. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes
3. Add tests for new functionality
4. Run tests and ensure they pass
5. Submit a pull request

### Code Style
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) guidelines
- Use type hints for all function signatures
- Add docstrings for all public functions and classes
- Keep functions small and focused on a single responsibility

## License
This project is licensed under the MIT License - see the LICENSE file for details.
