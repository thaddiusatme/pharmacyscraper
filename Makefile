.PHONY: help setup test lint format clean

# Define variables
PYTHON = python3
PIP = pip3
VENV = venv
PYTHON_VENV = $(VENV)/bin/python3
PIP_VENV = $(VENV)/bin/pip

# Help target
help:
	@echo "Available commands:"
	@echo "  make setup       - Set up the development environment"
	@echo "  make install     - Install the package in development mode"
	@echo "  make test        - Run tests"
	@echo "  make lint        - Check code style with flake8"
	@echo "  make format      - Format code with black"
	@echo "  make clean       - Remove build artifacts and cache"
	@echo "  make organize    - Organize existing data files"

# Set up the development environment
setup:
	@echo "Setting up development environment..."
	$(PYTHON) -m venv $(VENV)
	$(PIP_VENV) install --upgrade pip
	$(PIP_VENV) install -r requirements.txt
	$(PIP_VENV) install -e .
	@echo "\n✅ Setup complete! Activate the virtual environment with:\n   source $(VENV)/bin/activate"

# Install the package in development mode
install:
	$(PIP_VENV) install -e .


# Run tests
test:
	$(PYTHON_VENV) -m pytest tests/ -v

# Check code style
lint:
	$(PYTHON_VENV) -m flake8 scripts/ tests/
	$(PYTHON_VENV) -m mypy scripts/ tests/


# Format code
format:
	$(PYTHON_VENV) -m black scripts/ tests/


# Organize existing data files
organize:
	$(PYTHON_VENV) scripts/organize_data.py

# Clean up build artifacts
clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type d -name ".mypy_cache" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete

distclean: clean
	rm -rf $(VENV)
	rm -rf build/ dist/ *.egg-info/
	rm -f .coverage

# Help target
default: help
