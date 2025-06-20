# Development Guide

This guide provides information for developers working on the Independent Pharmacy Verification project.

## Table of Contents
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Code Style](#code-style)
- [Development Workflow](#development-workflow)
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
   pip install -r requirements-dev.txt
   
   # Install pre-commit hooks
   pre-commit install
   ```
4. **Configure environment variables**:
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

## Development Workflow

1. **Create a new branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-number-description
   ```

2. **Make your changes** following the code style guidelines.

3. **Run tests** to ensure nothing is broken:
   ```bash
   pytest tests/
   ```

4. **Update documentation** if your changes affect:
   - Command-line interfaces
   - Public APIs
   - Configuration options
   - Dependencies

5. **Commit your changes** with a descriptive message:
   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```

6. **Push your branch** to GitHub:
   ```bash
   git push -u origin your-branch-name
   ```

7. **Open a pull request** against the `main` branch.

## Testing

See the [TESTING.md](TESTING.md) guide for detailed information about running and writing tests.

## Documentation

- Keep the README up to date with any changes to setup or usage
- Update CHANGELOG.md for all user-visible changes
- Add docstrings for all public functions and classes
- Document any new environment variables in `.env.example`

## Pull Requests

When creating a pull request:

1. **Title** should be clear and descriptive
   - Use the format: `type(scope): description`
   - Example: `feat(collector): add retry logic for API calls`

2. **Description** should include:
   - Purpose of the changes
   - Any breaking changes
   - Related issues or PRs

3. **Checks** must pass:
   - All tests
   - Code coverage
   - Linting and type checking

## Release Process

1. **Update version numbers** in the appropriate files
2. **Update CHANGELOG.md** with release notes
3. **Create a release tag**:
   ```bash
   git tag -a v1.0.0 -m "Version 1.0.0"
   git push origin v1.0.0
   ```
4. **Create a GitHub release** with the changelog

## Getting Help

If you need help or have questions:
- Check the project documentation
- Search the issue tracker
- Open a new issue if needed