# 🚀 June 2025: Major Classification Pipeline Update

### Chain & Hospital Pre-filtering
The pipeline now pre-filters known chain and hospital/health system pharmacies before sending data to the Perplexity API. This:
- **Reduces API cost** by skipping obvious non-independents via local rules
- **Improves output accuracy**: Only ambiguous/non-chain/non-hospital pharmacies are classified by the LLM
- **Classification is now nearly error-free for all test batches**

#### Example Results
- **CA/WA/OR batch:** 230 pharmacies, 123 API calls, 77 chains, 27 hospitals pre-filtered
- **NY/TX/FL/IL/PA batch:** 387 pharmacies, 338 API calls, 60 chains pre-filtered

### How to Run State Batches
You can now process any set of states efficiently:

```bash
python3 scripts/process_cached_data.py --states ca,wa,or --output-dir data/processed_test_cawao
python3 scripts/process_cached_data.py --states ny,tx,fl,il,pa --output-dir data/processed_test_next5
```
The script will print summary stats for each location, showing how many chains/hospitals were skipped and how many required LLM classification.

### Classification Logic
- **Pre-filter:** Local keyword lists for chain and hospital/health system names
- **LLM step:** Only for ambiguous/unknown pharmacies
- **Post-processing:** Final output contains only true independents

### Why This Matters
- **Cost savings:** Most API calls are avoided for obvious chains/hospitals
- **Accuracy:** No hospital or chain pharmacies are misclassified as independent
- **Scalability:** Ready for full 50-state run

---
# Independent Pharmacy Verification Project

## Scalable Data Collection Pipeline

This project has been enhanced with a robust, end-to-end pipeline for collecting and processing pharmacy data at scale. The new workflow automates collection, deduplication, and classification, ensuring data quality and consistency.

### Key Features

- **End-to-End Automation**: The `scripts/run_pipeline.py` script now orchestrates the entire data processing workflow, from initial data collection via Apify to final classification. This replaces the previous manual, multi-step process.

- **Scalable Configuration**: A new `config/large_scale_run.json` file has been created to manage a comprehensive, 50-state data collection effort. This configuration is designed to gather approximately 2,500 pharmacy records.

- **Robust and Safe Execution**:
  - **Persistent Logging**: The pipeline now supports file-based logging via the `--log-file` argument, ensuring a complete record of all actions and errors during long-running jobs.
  - **Uninterrupted Operation**: For macOS users, the `caffeinate` command is used to prevent the system from sleeping during execution, safeguarding against data loss or corruption.

- **Phased Rollout Strategy**: To manage costs and validate the pipeline's performance, a smaller `config/five_state_run.json` is used for initial test runs. This allows for verification of the entire process on a smaller dataset before launching the full 50-state collection.

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
