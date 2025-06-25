# Contributing to Pharmacy Scraper

Thank you for your interest in contributing to the Pharmacy Scraper project! This document will help you get started with contributing.

## Development Setup

1. **Fork the repository**
   - Click the "Fork" button at the top-right of the repository page

2. **Clone your fork**
   ```bash
   git clone https://github.com/your-username/pharmacy-scraper.git
   cd pharmacy-scraper
   ```

3. **Set up a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install dependencies**
   ```bash
   pip install -e .[dev]
   ```

## Development Workflow

1. **Create a new branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the existing code style
   - Add tests for new features
   - Update documentation as needed

3. **Run tests**
   ```bash
   pytest tests/ -v
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add your feature"
   ```

5. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request**
   - Go to the original repository
   - Click "New Pull Request"
   - Select your fork and branch
   - Fill in the PR template
   - Submit the PR

## Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code
- Use type hints for all function signatures
- Keep functions small and focused on a single task
- Write docstrings for all public functions and classes

## Testing

- Write tests for new features
- Run all tests before submitting a PR
- Aim for good test coverage

## Documentation

- Update documentation when adding new features
- Keep docstrings up to date
- Add examples for complex functionality

## Reporting Issues

When reporting issues, please include:
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Any relevant error messages
- Python version and operating system

## Code Review Process

1. A maintainer will review your PR
2. You may be asked to make changes
3. Once approved, a maintainer will merge your PR

Thank you for contributing!
