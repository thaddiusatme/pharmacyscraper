#!/usr/bin/env python3
"""
Config validation helper for pharmacy scraper configurations.
Validates JSON config files against the schema.

Usage:
    python scripts/validate_config.py config/example_config.json
    python scripts/validate_config.py config/production/secure_production_config.json
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import jsonschema
    from jsonschema import validate, ValidationError
except ImportError:
    print("jsonschema not installed. Install it with: pip install jsonschema")
    sys.exit(1)


def load_schema() -> dict:
    """Load the config schema from config/schema.json"""
    schema_path = Path(__file__).parent.parent / "config" / "schema.json"
    try:
        with open(schema_path) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Schema file not found: {schema_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in schema file: {e}")
        sys.exit(1)


def validate_config(config_path: str) -> tuple[bool, list[str]]:
    """
    Validate a config file against the schema.
    
    Args:
        config_path: Path to the config file to validate
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    schema = load_schema()
    
    try:
        with open(config_path) as f:
            config = json.load(f)
    except FileNotFoundError:
        return False, [f"Config file not found: {config_path}"]
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON in config file: {e}"]
    
    try:
        validate(instance=config, schema=schema)
        return True, []
    except ValidationError as e:
        error_msg = f"Validation error at {'.'.join(str(p) for p in e.path)}: {e.message}"
        return False, [error_msg]
    except Exception as e:
        return False, [f"Unexpected validation error: {e}"]


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Validate pharmacy scraper config files")
    parser.add_argument("config_path", help="Path to config file to validate")
    parser.add_argument("--quiet", "-q", action="store_true", help="Only show errors")
    
    args = parser.parse_args()
    
    if not args.quiet:
        print(f"Validating config: {args.config_path}")
    
    is_valid, errors = validate_config(args.config_path)
    
    if is_valid:
        if not args.quiet:
            print("✅ Config is valid!")
        sys.exit(0)
    else:
        print("❌ Config validation failed:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
