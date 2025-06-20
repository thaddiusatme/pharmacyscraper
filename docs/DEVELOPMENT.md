# Development Guide

This guide provides information for developers working on the Independent Pharmacy Verification project.

## Table of Contents
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Code Style](#code-style)
- [Development Workflow](#development-workflow)
- [API Integration](#api-integration)
- [Credit Management](#credit-management)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Requests](#pull-requests)
- [Release Process](#release-process)

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone git@github.com:your-username/pharmacy-verification.git
   cd pharmacy-verification
   ```
3. **Set up the development environment**:
   ```bash
   # Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   
   # Install development dependencies
   pip install -r requirements.txt
   ```
4. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

## Project Structure

```
.
├── data/               # Data files (not versioned)
│   ├── raw/            # Raw collected data
│   └── processed/      # Processed and cleaned data
├── docs/               # Documentation
│   ├── DEVELOPMENT.md  # This file
│   └── TESTING.md      # Testing guide
├── scripts/            # Python modules
│   ├── apify_collector.py  # Apify data collection
│   └── organize_data.py    # Data organization utilities
├── tests/              # Test suite
│   ├── conftest.py     # Test fixtures
│   └── test_*.py       # Test files
├── .env.example        # Example environment variables
├── .gitignore          # Git ignore rules
├── .pre-commit-config.yaml  # Pre-commit hooks
├── CHANGELOG.md        # Project changelog
├── LICENSE             # License file
├── README.md           # Project README
└── requirements.txt    # Project dependencies
```

## Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Use type hints for all function signatures
- Use Google-style docstrings
- Keep lines under 88 characters (Black's default)
- Use `isort` for import sorting
- Use `black` for code formatting

### Pre-commit Hooks

This project uses pre-commit hooks to ensure code quality. The following hooks are configured:
- `black`: Code formatting
- `isort`: Import sorting
- `flake8`: Linting
- `mypy`: Static type checking

These run automatically on each commit. You can also run them manually:

```bash
pre-commit run --all-files
```

## API Integration

### Apify Google Maps Scraper

The project uses Apify's Google Maps Scraper for collecting pharmacy data. Key components:

- **Location**: `src/dedup_self_heal/apify_integration.py`
- **Configuration**: Set in `.env` file
  - `APIFY_API_TOKEN`: Your Apify API token
  - `API_BUDGET`: Total credit budget (default: 100.0)
  - `API_DAILY_LIMIT`: Daily credit limit (default: 25.0)

#### Features:
- Rate limiting (10 requests/minute by default)
- Credit tracking and budget enforcement
- Automatic retries with exponential backoff
- Error handling and logging

### Running the Integration

```python
from src.dedup_self_heal.apify_integration import ApifyPharmacyScraper

# Initialize with API token from environment
scraper = ApifyPharmacyScraper()

# Search for pharmacies
results = scraper.scrape_pharmacies(
    state="CA",
    city="San Francisco",
    max_results=5
)
```

## Credit Management

### Overview
The project implements a credit-based system to track and limit API usage across different services.

### Configuration
Set these in your `.env` file:
```
API_BUDGET=100.0
API_DAILY_LIMIT=25.0
```

### Usage
```python
from src.utils.api_usage_tracker import credit_tracker

# Check available credits
if credit_tracker.check_credit_available(1.0):
    # Make API call
    credit_tracker.record_usage(1.0, "API Call Description")
```

### Features
- Tracks total and daily credit usage
- Enforces budget limits
- Persists usage data between runs
- Provides usage statistics

## Development Workflow

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Implement your changes** following TDD:
   - Write a failing test
   - Implement the minimum code to pass
   - Refactor as needed

3. **Run tests**:
   ```bash
   pytest tests/
   ```

4. **Update documentation** as needed

5. **Commit your changes** with a descriptive message

6. **Push and create a pull request**

## Testing

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_module.py

# Run with coverage report
pytest --cov=src tests/
```

### Writing Tests
- Place test files in the `tests/` directory
- Name test files with `test_` prefix
- Use descriptive test function names starting with `test_`
- Use fixtures for common test data

## Documentation

### Updating Documentation
- Update `CHANGELOG.md` for all notable changes
- Keep `README.md` up-to-date with setup instructions
- Document new features in relevant `.md` files
- Add docstrings to all public functions and classes

## Pull Requests

1. Keep PRs focused on a single feature/bugfix
2. Include tests for new functionality
3. Update documentation as needed
4. Request review from at least one team member

## Release Process

1. Update `CHANGELOG.md` with release notes
2. Update version in `__init__.py`
3. Create a release tag:
   ```bash
   git tag -a v1.0.0 -m "Version 1.0.0"
   git push origin v1.0.0
   ```
4. Create a GitHub release with release notes