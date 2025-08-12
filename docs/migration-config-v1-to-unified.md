# Migration Guide: Config V1 -> Unified Schema

This guide helps migrate existing JSON configs to the new unified schema that supports JSON or YAML, environment inheritance, and stricter validation.

## Key Changes
- JSON or YAML config supported (file extension determines parser)
- Optional environment inheritance via `env` and `environments`
- Stricter validation with helpful errors
- Optional unified fields for higher-level config: `search_terms`, `regions`

## Minimal Example (JSON)
```json
{
  "api_keys": {
    "google_places": "${GOOGLE_PLACES_KEY}",
    "perplexity": "${PERPLEXITY_API_KEY}"
  },
  "output_dir": "output",
  "cache_dir": "cache",
  "locations": ["San Francisco, CA"],
  "max_results_per_query": 25
}
```

## Minimal Example (YAML)
```yaml
api_keys:
  google_places: ${GOOGLE_PLACES_KEY}
  perplexity: ${PERPLEXITY_API_KEY}
output_dir: output
cache_dir: cache
locations:
  - San Francisco, CA
max_results_per_query: 25
```

## Using `env` and `environments`
```yaml
env: prod
environments:
  prod:
    max_budget: 200
    api_cost_limits:
      apify: 0.6
  dev:
    max_budget: 50
```

## Unified Fields (Optional)
- `search_terms`: list of strings (e.g., ["pharmacy", "compounding pharmacy"]) 
- `regions`: list of strings (e.g., ["CA", "NYC"]) or list of dicts with minimal structure, like:
```yaml
regions:
  - { state: CA }
  - { city: "San Francisco" }
  - { name: "Bay Area" }
```

## Validation Rules (Highlights)
- `output_dir`, `cache_dir`: strings
- `classification_threshold`, `verification_confidence_threshold`: 0.0–1.0 if present
- `max_budget`: non-negative number if present
- `api_cost_limits`: dict of non-negative numbers
- `locations`: list of strings if present
- `search_terms`: non-empty list of strings if present
- `regions`: list of strings or dicts containing at least one of: `state`, `city`, `name`

## Migration Tips
- Start by renaming your existing JSON file to `.yaml` if you prefer YAML; otherwise keep JSON.
- Add `env` and `environments` if you need per-environment overrides.
- Introduce `search_terms` and `regions` gradually; existing `locations` continue to work.
- Use environment substitution `${VAR}` or `${VAR:-default}` to avoid committing secrets.

## FAQ
- Q: Do I have to switch to YAML?  
  A: No. Both JSON and YAML are supported.
- Q: Are old configs still valid?  
  A: Yes, as long as they pass the stricter top-level key allowlist and basic type checks.
- Q: Is there a breaking change?  
  A: No breaking changes expected; this is additive and validation-only.
