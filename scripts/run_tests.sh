#!/bin/bash
# Run the test suite for the pharmacy scraper

# Exit on error
set -e

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Install test requirements if not already installed
pip install -q pytest pytest-cov pytest-mock pandas

# Run pytest with coverage and detailed output
echo "🚀 Running tests with coverage..."
python -m pytest tests/ -v \
    --cov=scripts \
    --cov-report=term-missing:skip-covered \
    --cov-report=html:htmlcov \
    --cov-fail-under=80

# Check if tests passed
if [ $? -eq 0 ]; then
    echo -e "\n✅ All tests passed!"
    
    # Show coverage summary
    echo -e "\n📊 Coverage Summary:"
    coverage report --skip-covered --fail-under=80
    
    # Open coverage report in browser
    if command -v open &> /dev/null; then
        echo -e "\n🌐 Opening HTML coverage report..."
        open htmlcov/index.html
    fi
else
    echo -e "\n❌ Some tests failed"
    exit 1
fi
