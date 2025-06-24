import json
import os
import csv
import argparse
from datetime import datetime

def generate_spreadsheet_report(data_dirs, output_path):
    """
    Scans specified data directories, extracts detailed pharmacy data from JSON files,
    and generates a single CSV spreadsheet.
    """
    all_pharmacies = []

    for data_dir in data_dirs:
        print(f"Scanning directory: {data_dir}")
        try:
            files_in_dir = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                            if f.endswith('.json') and 'summary' not in f and 'filter' not in f]
            for file_path in files_in_dir:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        # Each file can contain a list of pharmacies
                        for pharmacy in data:
                            all_pharmacies.append(pharmacy)
                except (json.JSONDecodeError, IOError) as e:
                    print(f"Warning: Could not read or parse {os.path.basename(file_path)}. Skipping. Error: {e}")
        except FileNotFoundError:
            print(f"Warning: Directory not found, skipping: {data_dir}")
            continue

    if not all_pharmacies:
        print("No pharmacy data found to generate a spreadsheet.")
        return

    # Define the headers for the CSV file
    headers = [
        'title', 'address', 'city', 'state', 'postalCode', 'phone', 'website',
        'classification', 'totalScore', 'reviewsCount', 'categoryName',
        'neighborhood', 'street', 'countryCode', 'permanentlyClosed',
        'temporarilyClosed', 'placeId', 'cid', 'scrapedAt'
    ]

    print(f"\nFound a total of {len(all_pharmacies)} pharmacy records.")

    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(all_pharmacies)
        print(f"\n✅ Spreadsheet successfully saved to {output_path}")
    except IOError as e:
        print(f"\n❌ Error saving spreadsheet to file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a consolidated pharmacy data spreadsheet from multiple data directories.")
    parser.add_argument(
        '--data-dirs',
        nargs='+',
        required=True,
        help='One or more directories containing the processed JSON files.'
    )
    
    # Generate a default filename with a timestamp to avoid overwriting
    default_filename = f"pharmacy_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    parser.add_argument(
        '--output-path',
        default=default_filename,
        help=f'Path to save the final CSV spreadsheet. Defaults to {default_filename}.'
    )
    args = parser.parse_args()

    generate_spreadsheet_report(args.data_dirs, args.output_path)
