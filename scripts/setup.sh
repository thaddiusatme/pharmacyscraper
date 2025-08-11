#!/bin/bash

# Exit on error
set -e

echo "🚀 Setting up Pharmacy Verification Project..."

# Create virtual environment
echo "🐍 Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "🔄 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Create directory structure
echo "📂 Creating project directory structure..."
mkdir -p data/raw data/processed
mkdir -p scripts
mkdir -p logs
mkdir -p reports
mkdir -p docs

# Set up environment variables file
echo "🔑 Creating .env file for environment variables..."
cat > .env <<EOL
# Google Maps API Key
GOOGLE_MAPS_API_KEY=your_google_maps_api_key_here

# Apify API Token (if using Apify)
APIFY_API_TOKEN=your_apify_api_token_here
EOL

# Make setup script executable
chmod +x setup.sh

# Create initial README.md
cat > README.md << 'EOL'
# Independent Pharmacy Verification Project

## Project Overview
This project collects and verifies information about independent pharmacies across the United States.

## Setup

1. Clone the repository
   ```bash
   git clone <repository-url>
   cd pharmacy-verification
   ```

2. Set up the environment
   ```bash
   # Run the setup script
   ./setup.sh
   
   # Activate virtual environment
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. Configure environment variables
   - Copy `.env.example` to `.env`
   - Add your API keys and configuration

## Project Structure

```
.
├── data/               # Data files
│   ├── raw/            # Raw data from sources
│   └── processed/      # Processed and cleaned data
├── scripts/            # Python scripts
├── logs/               # Log files
├── reports/            # Generated reports
├── docs/               # Documentation
├── .gitignore
├── requirements.txt
└── README.md
```

## Usage

1. Run data collection:
   ```bash
   python scripts/collect_pharmacies.py
   ```

2. Run verification:
   ```bash
   python scripts/verify_pharmacies.py
   ```

3. Generate reports:
   ```bash
   python scripts/generate_report.py
   ```

## License
MIT
EOL

echo "✅ Setup complete! Don't forget to:"
echo "1. Add your API keys to the .env file"
echo "2. Activate the virtual environment with 'source venv/bin/activate'"
echo "3. Run 'pre-commit install' if you want to use pre-commit hooks"
