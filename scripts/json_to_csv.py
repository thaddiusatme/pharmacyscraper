#!/usr/bin/env python3
"""
Convert pharmacy data from JSON to CSV format.
"""
import json
import csv
import os
from pathlib import Path

def convert_json_to_csv(json_path, output_dir=None):
    """Convert JSON file to CSV format.
    
    Args:
        json_path: Path to the input JSON file
        output_dir: Directory to save the output CSV (default: same as JSON file)
    """
    json_path = Path(json_path)
    if output_dir is None:
        output_dir = json_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename
    csv_path = output_dir / f"{json_path.stem}.csv"
    
    # Read JSON data
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not data:
        print(f"No data found in {json_path}")
        return
    
    # Extract all possible field names from the data
    fieldnames = set()
    for item in data:
        fieldnames.update(item.keys())
    
    # Flatten nested structures
    flat_data = []
    for item in data:
        flat_item = {}
        for key, value in item.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    flat_item[f"{key}_{subkey}"] = subvalue
            elif isinstance(value, list):
                flat_item[key] = ", ".join(str(v) for v in value)
            else:
                flat_item[key] = value
        flat_data.append(flat_item)
    
    # Update fieldnames with flattened fields
    fieldnames = set()
    for item in flat_data:
        fieldnames.update(item.keys())
    fieldnames = sorted(fieldnames)
    
    # Write to CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(flat_data)
    
    print(f"Successfully converted {len(data)} records to {csv_path}")
    return csv_path

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python json_to_csv.py <input_json> [output_dir]")
        sys.exit(1)
    
    input_json = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        convert_json_to_csv(input_json, output_dir)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
