#!/usr/bin/env python3
"""
Validate Cascade rules YAML file.

This script checks that the Cascade rules YAML file has the required structure.
"""

import os
import sys
import yaml

def validate_yaml(file_path):
    """Validate a YAML file for required Cascade rules structure."""
    try:
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        
        print(f"✓ Valid YAML: {file_path}")
        print(f"  Contains keys: {list(data.keys()) if data else 'empty'}")
        
        # Check for required fields
        required_fields = ["rules", "version"]
        missing = [field for field in required_fields if field not in data]
        
        if missing:
            print(f"❌ Missing required fields: {', '.join(missing)}")
            return False
            
        print("✓ All required fields present")
        return True
        
    except Exception as e:
        print(f"❌ Error in {file_path}: {str(e)}")
        return False

def verify_pharmacy_scraper(file_path):
    """Verify the pharmacy_scraper.yaml file specifically."""
    try:
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        print(f"✓ Verified pharmacy_scraper.yaml")
        print(f"  Version: {data.get('version', 'Not specified')}")
        print(f"  Number of rules: {len(data.get('rules', []))}")
        return True
    except Exception as e:
        print(f"❌ Error verifying {file_path}: {str(e)}")
        return False

def main():
    """Main function to validate Cascade rules."""
    if len(sys.argv) < 2:
        print("Usage: python validate_cascade.py <yaml_file> [--verify]")
        sys.exit(1)
        
    file_path = sys.argv[1]
    is_verify = len(sys.argv) > 2 and sys.argv[2] == "--verify"
    
    if not os.path.isfile(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    if is_verify:
        if not verify_pharmacy_scraper(file_path):
            sys.exit(1)
    else:
        if not validate_yaml(file_path):
            sys.exit(1)
    
    sys.exit(0)

if __name__ == "__main__":
    main()
