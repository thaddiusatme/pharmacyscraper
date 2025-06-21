# Independent Pharmacy Verification Project

## Recent Updates (June 2024)

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
This project collects and verifies information about independent pharmacies across the United States using Apify's Google Maps Scraper.

## Features
- Automated data collection from Google Maps via Apify
- Comprehensive test suite with proper mocking and isolation
- Support for both simple queries and structured location-based searches
- Data validation and cleaning
- Chain pharmacy filtering
- CSV export functionality

## 🚀 Features

- **Automated Data Collection**
  - Apify-based Google Maps scraping
  - Structured location-based queries
  - Chain pharmacy filtering
  - Rate limiting and error handling

- **AI-Powered Classification**
  - Hybrid LLM/rule-based classification
  - Perplexity API integration
  - Caching for performance
  - 83% test coverage

- **Address Verification**
  - Google Places API integration
  - 90% verification success rate
  - Confidence scoring
  - Duplicate detection

- **Data Pipeline**
  - End-to-end processing
  - Configurable workflows
  - Comprehensive logging
  - Test coverage tracking

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
pytest --cov=scripts tests/
```

## Project Structure

```
.
├── data/               # Data files
│   ├── raw/            # Raw data from sources
│   └── processed/      # Processed and cleaned data
├── scripts/            # Python scripts
│   ├── apify_collector.py  # Apify data collection
│   └── organize_data.py    # Data organization utilities
├── tests/              # Test files
│   ├── conftest.py     # Test fixtures
│   └── test_apify_collector.py  # Apify collector tests
├── logs/               # Log files
├── reports/            # Generated reports
├── .env.example        # Example environment variables
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Usage

1. Run data collection:
   ```bash
   python -m scripts.apify_collector
   ```
   
   This will:
   - Generate search queries for independent pharmacies
   - Collect data using Apify's Google Maps Scraper
   - Save results to `data/raw/`
   - Generate a combined and deduplicated file in `data/processed/`

2. View the collected data:
   ```bash
   # View the processed data
   python -c "import pandas as pd; print(pd.read_csv('data/processed/combined_pharmacies.csv').head())"
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
