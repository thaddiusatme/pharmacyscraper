# Independent Pharmacy Verification Project

## Project Overview
This project collects and verifies information about independent pharmacies across the United States using Apify's Google Maps Scraper.

## Features
- Automated data collection from Google Maps via Apify
- Test suite with comprehensive test coverage
- Data validation and cleaning
- Chain pharmacy filtering
- CSV export functionality

## Setup

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
