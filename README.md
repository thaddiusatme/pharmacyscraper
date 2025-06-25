# Pharmacy Scraper

A Python-based tool for scraping and analyzing pharmacy data from various sources, with a focus on identifying independent, non-hospital pharmacies.

## Features

- **Data Collection**: Scrapes pharmacy data using Apify and Google Places API
- **Filtering**: Advanced filtering to identify independent pharmacies
- **Verification**: Address and business verification using Google Places
- **Deduplication**: Smart duplicate removal and self-healing capabilities
- **Classification**: AI-powered classification of pharmacies
- **Budget Management**: Tracks and manages API usage and costs

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
