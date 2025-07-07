#!/bin/bash
# Pharmacy Scraper Environment Setup and Runner Script
# This script securely loads API keys from environment variables and runs the pipeline

# Ensure script exits on first error
set -e

# Directory for environment file
ENV_FILE=".env"

# Check if .env file exists, if not create template
if [ ! -f $ENV_FILE ]; then
    echo "Creating template .env file. Please fill in your API keys."
    cat > $ENV_FILE << EOL
# Pharmacy Scraper API Keys
# WARNING: Never commit this file to version control!
APIFY_TOKEN=your_apify_token_here
GOOGLE_PLACES_API_KEY=your_google_places_api_key_here
PERPLEXITY_API_KEY=your_perplexity_api_key_here
EOL
    echo "Created $ENV_FILE template. Edit this file to add your API keys before proceeding."
    exit 1
fi

# Source environment variables from .env file
source $ENV_FILE

# Check that required environment variables are set
if [ -z "$APIFY_TOKEN" ] || [ -z "$GOOGLE_PLACES_API_KEY" ] || [ -z "$PERPLEXITY_API_KEY" ]; then
    echo "Error: Missing required API keys in $ENV_FILE"
    echo "Please ensure all API keys are properly set in $ENV_FILE"
    exit 1
fi

# Create necessary directories
mkdir -p cache/production output/production

# Process command line arguments
CONFIG_PATH="config/production/secure_production_config.json"
RESET_FLAG=""

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --config)
        CONFIG_PATH="$2"
        shift
        shift
        ;;
        --reset)
        RESET_FLAG="--reset"
        shift
        ;;
        *)
        echo "Unknown option: $1"
        echo "Usage: $0 [--config CONFIG_PATH] [--reset]"
        exit 1
        ;;
    esac
done

# Run the pipeline with the provided config
echo "Starting pharmacy scraper pipeline with configuration: $CONFIG_PATH"
python -m src.pharmacy_scraper.run_pipeline_v2 --config "$CONFIG_PATH" $RESET_FLAG

# Check exit status
if [ $? -eq 0 ]; then
    echo "✅ Pipeline completed successfully!"
else
    echo "❌ Pipeline encountered errors. Check the logs for details."
    exit 1
fi
