# Pharmacy Scraper Classification Module

This module provides AI-powered classification capabilities for identifying independent pharmacies using a combination of rule-based classification and LLM-based approaches.

## Components

### 1. Classifier

The main entry point for pharmacy classification that implements a hybrid approach:
- Rule-based classification for clear cases
- LLM-based classification for ambiguous cases
- Intelligent caching to minimize API costs

### 2. PerplexityClient

Provides integration with Perplexity's AI API for pharmacy classification.

#### Model Information

As of July 2025, the classification system uses Perplexity's `sonar` model:

- **Model**: `sonar`
- **Context Length**: 128k tokens
- **Features**: Lightweight search model with grounding capabilities
- **Use Case**: Optimized for classification tasks with web search capability
- **Previous Model**: `pplx-7b-chat` (deprecated)

#### Configuration

The Perplexity client can be configured with the following parameters:

```python
client = PerplexityClient(
    api_key="your_api_key",  # If None, will use PERPLEXITY_API_KEY env var
    model_name="sonar",      # Default model
    temperature=0.1,         # Low temperature for consistent results
    cache_enabled=True,      # Enable caching to reduce API costs
    system_prompt="You are a pharmacy classification expert..."
)
```

#### Security

API keys are handled securely through:
1. Environment variables (recommended)
2. Secure configuration files
3. Direct initialization (not recommended for production)

Never hardcode API keys in source code or commit them to version control.

#### Budget Management

The client integrates with the API credit tracker system to enforce budget limits:

```python
# Configuration example
{
  "max_budget": 50.0,
  "api_cost_limits": {
    "perplexity": 0.2  # Cost per API call
  }
}
```

## Usage Examples

### Basic Classification

```python
from pharmacy_scraper.classification import Classifier
from pharmacy_scraper.classification.data_models import PharmacyData

# Create a classifier
classifier = Classifier(perplexity_api_key="YOUR_API_KEY")

# Define a pharmacy
pharmacy = PharmacyData(
    name="Community Corner Pharmacy",
    address="123 Main St",
    city="Phoenix",
    state="AZ",
    phone="555-123-4567"
)

# Classify the pharmacy
result = classifier.classify_pharmacy(pharmacy)

# Check if it's an independent pharmacy
if result.is_independent:
    print(f"{pharmacy.name} is an independent pharmacy (confidence: {result.confidence})")
else:
    print(f"{pharmacy.name} is not an independent pharmacy")
```

### Batch Classification

```python
# For multiple pharmacies
pharmacies = [pharmacy1, pharmacy2, pharmacy3]
results = classifier.classify_pharmacies(pharmacies)

for pharmacy, result in zip(pharmacies, results):
    print(f"{pharmacy.name}: Independent={result.is_independent}")
```

## Error Handling

The classification module implements comprehensive error handling:

- API rate limiting
- Network errors
- Model availability issues
- Response parsing failures

All exceptions are properly logged and categorized for debugging.

## Testing

The classification module has extensive test coverage:
- 100% coverage for the Classifier class
- 92% coverage for the PerplexityClient

Tests use mocking to avoid actual API calls, making them safe to run in CI/CD pipelines.
