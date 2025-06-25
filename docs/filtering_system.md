# Pharmacy Filtering System

## Overview
The pharmacy filtering system provides centralized, configurable rules for identifying and filtering out chain pharmacies, hospital-affiliated pharmacies, and other unwanted results. This ensures that only relevant independent pharmacies are included in the final output.

## Key Components

### 1. Centralized Filter Module (`src/pharmacy_scraper/config/filters.py`)

Contains comprehensive lists and functions for filtering:

- **Chain Pharmacies**: Major retail chains (CVS, Walgreens, etc.)
- **Hospital Terms**: Hospital and healthcare system indicators
- **Exclusion Terms**: Other terms that indicate non-independent pharmacies

### 2. Configuration File (`config/production/independent_pharmacies_config.json`)

Main configuration that specifies:
- Target cities and queries
- Filtering rules and thresholds
- API and budget settings
- Output format

## Usage

### Basic Filtering

```python
from pharmacy_scraper.config.filters import filter_pharmacy

# Example pharmacy data
pharmacy = {
    "name": "CVS Pharmacy #123",
    "address": "123 Main St",
    "city": "Phoenix",
    "state": "AZ"
}

# Check if pharmacy should be included
if filter_pharmacy(pharmacy):
    print("Include in results")
else:
    print("Exclude from results")
```

### Configuration Options

Key settings in the configuration file:

```json
{
  "filtering": {
    "use_centralized_filters": true,
    "filter_module": "pharmacy_scraper.config.filters",
    "min_rating": 3.5,
    "verify_address": true,
    "require_phone": true
  }
}
```

### Adding New Filter Terms

To add new terms to any filter set, edit the appropriate constant in `filters.py`:

```python
# In src/pharmacy_scraper/config/filters.py

# Add to chain pharmacies
CHAIN_PHARMACIES.update({"New Chain Pharmacy", "Another Chain"})

# Add to hospital terms
HOSPITAL_TERMS.update({"Medical Group", "Healthcare System"})

# Add to exclusion terms
EXCLUSION_TERMS.update({"Specialty Center", "Research Facility"})
```

## Testing

Run the test suite to verify the filtering logic:

```bash
pytest tests/test_filters.py -v
```

## Best Practices

1. **Use Centralized Filters**: Always use the centralized filter module rather than hardcoding filter logic.
2. **Regular Updates**: Periodically review and update the filter terms based on new data.
3. **Test Changes**: Run tests after modifying filter terms to ensure they work as expected.
4. **Monitor Results**: Regularly review filtered results to identify any false positives/negatives.

## Troubleshooting

### Common Issues

1. **False Positives**: If legitimate pharmacies are being filtered out, check if their names contain any terms in the exclusion lists.
2. **False Negatives**: If chain/hospital pharmacies are not being filtered, add appropriate terms to the relevant filter sets.
3. **Performance**: For large datasets, ensure the filter module is imported only once and reused.

### Debugging

To debug filtering decisions, you can use the individual filter functions:

```python
from pharmacy_scraper.config.filters import (
    is_chain_pharmacy,
    is_hospital_pharmacy,
    should_exclude
)

pharmacy_name = "Mayo Clinic Pharmacy"
print(f"Is chain: {is_chain_pharmacy(pharmacy_name)}")
print(f"Is hospital: {is_hospital_pharmacy(pharmacy_name)}")
print(f"Should exclude: {should_exclude(pharmacy_name)}")
```

## Maintenance

- Review and update filter terms quarterly
- Monitor for new pharmacy chains or healthcare systems
- Update tests when adding new filter terms
