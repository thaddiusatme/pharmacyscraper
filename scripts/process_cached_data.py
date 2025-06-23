#!/usr/bin/env python3
"""
Process cached Apify data from 50-state run and run classification
"""


import json
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List
import sys
import argparse
import logging
import traceback

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.classification.classifier import Classifier
from src.utils.api_usage_tracker import APICreditTracker

def load_cached_data(cache_dir: str = ".api_cache/apify", limit: int = None, skip_existing: bool = False, output_dir: str = "data/processed_50_state") -> Dict[str, List[Dict]]:
    """Load cached pharmacy data from Apify, optionally limited to first N files"""
    cached_data = {}
    cache_path = Path(cache_dir)
    output_path = Path(output_dir)
    print(f"Scanning cache directory: {cache_path}")
    if not cache_path.exists():
        print(f"Cache directory {cache_path} does not exist")
        return cached_data
    json_files = list(cache_path.glob("*.json"))
    print(f"Found {len(json_files)} cached files")

    # Filter out already processed files if skip_existing is set
    unprocessed_files = []
    for file_path in json_files:
        filename = file_path.stem
        if filename.startswith("independent_pharmacy_"):
            location = filename.replace("independent_pharmacy_", "")
            processed_file = output_path / f"pharmacies_{location}.json"
            if skip_existing and processed_file.exists():
                print(f"Skipping already processed file: {processed_file}")
                continue
        unprocessed_files.append(file_path)

    # Apply limit after filtering
    if limit:
        unprocessed_files = unprocessed_files[:limit]
        print(f"Processing first {len(unprocessed_files)} unprocessed files (limit={limit})")

    for file_path in unprocessed_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            # Extract query info from filename
            filename = file_path.stem  # e.g., "independent_pharmacy_houston_tx"
            if filename.startswith("independent_pharmacy_"):
                location = filename.replace("independent_pharmacy_", "")
                # If skip_existing is set, check if processed file exists
                if skip_existing:
                    processed_file = output_path / f"pharmacies_{location}.json"
                    if processed_file.exists():
                        continue
                cached_data[location] = data
                print(f"Loaded {len(data)} pharmacies for {location}")
        except Exception as e:
            err_msg = f"Error loading {file_path}: {e}\n{traceback.format_exc()}"
            print(f"[ERROR] {err_msg}")
            logging.error(err_msg)
    return cached_data

def process_and_classify_data(cached_data: Dict[str, List[Dict]], output_dir: str = "data/processed_50_state"):
    """Process cached data and run classification"""
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize classifier
    classifier = Classifier()
    
    # Track statistics
    total_pharmacies = 0
    total_classified = 0
    results_by_location = {}
    
    print("\n" + "="*60)
    print("PROCESSING AND CLASSIFYING CACHED DATA")
    print("="*60)
    
    for location, pharmacies in cached_data.items():
        print(f"\nProcessing {location}: {len(pharmacies)} pharmacies")
        
        classified_pharmacies = []
        
        for i, pharmacy in enumerate(pharmacies):
            try:
                # Convert to format expected by classifier
                pharmacy_dict = {
                    'name': pharmacy.get('title', ''),
                    'address': pharmacy.get('address', ''),
                    'phone': pharmacy.get('phone', ''),
                    'website': pharmacy.get('website', ''),
                    'description': pharmacy.get('description', ''),
                    'categoryName': pharmacy.get('categoryName', ''),
                    'location': {
                        'lat': pharmacy.get('location', {}).get('lat') if isinstance(pharmacy.get('location'), dict) else None,
                        'lng': pharmacy.get('location', {}).get('lng') if isinstance(pharmacy.get('location'), dict) else None
                    }
                }
                
                # Classify pharmacy
                print(f"  Classifying {i+1}/{len(pharmacies)}: {pharmacy_dict['name'][:50]}...")
                classification_result = classifier.classify_pharmacy(pharmacy_dict)
                
                # Add classification to pharmacy data
                pharmacy_with_classification = pharmacy.copy()
                pharmacy_with_classification.update(classification_result)
                
                classified_pharmacies.append(pharmacy_with_classification)
                total_classified += 1
                
            except Exception as e:
                err_msg = f"Failed to classify {pharmacy.get('title', '')}: {e}\n{traceback.format_exc()}"
                print(f"  [ERROR] {err_msg}")
                logging.error(err_msg)
                # Add original pharmacy without classification
                classified_pharmacies.append(pharmacy)
            
            total_pharmacies += 1
        
        # Save results for this location
        location_file = output_path / f"pharmacies_{location}.json"
        with open(location_file, 'w') as f:
            json.dump(classified_pharmacies, f, indent=2)
        
        # Calculate stats for this location
        independent_count = sum(1 for p in classified_pharmacies 
                              if p.get('classification') == 'independent')
        chain_count = sum(1 for p in classified_pharmacies 
                         if p.get('classification') == 'chain')
        error_count = sum(1 for p in classified_pharmacies 
                         if p.get('classification') in ['error', None])
        
        results_by_location[location] = {
            'total': len(classified_pharmacies),
            'independent': independent_count,
            'chain': chain_count,
            'errors': error_count
        }
        
        print(f"    ✅ Saved to {location_file}")
        print(f"    📊 Independent: {independent_count}, Chain: {chain_count}, Errors: {error_count}")
    
    # Create summary report
    create_summary_report(results_by_location, output_path, total_pharmacies, total_classified)
    
    return results_by_location

def create_summary_report(results_by_location: Dict, output_path: Path, total_pharmacies: int, total_classified: int):
    """Create a summary report of the processing results"""
    
    summary_file = output_path / "processing_summary.json"
    
    # Calculate totals
    total_independent = sum(r['independent'] for r in results_by_location.values())
    total_chain = sum(r['chain'] for r in results_by_location.values())
    total_errors = sum(r['errors'] for r in results_by_location.values())
    
    summary = {
        "total_locations_processed": len(results_by_location),
        "total_pharmacies": total_pharmacies,
        "total_classified": total_classified,
        "classification_success_rate": f"{(total_classified/total_pharmacies)*100:.1f}%" if total_pharmacies > 0 else "0%",
        "totals": {
            "independent": total_independent,
            "chain": total_chain,
            "errors": total_errors
        },
        "by_location": results_by_location
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"📍 Locations processed: {len(results_by_location)}")
    print(f"🏥 Total pharmacies: {total_pharmacies}")
    print(f"✅ Successfully classified: {total_classified}")
    print(f"📊 Independent: {total_independent}")
    print(f"🏪 Chain: {total_chain}")
    print(f"❌ Errors: {total_errors}")
    print(f"💯 Success rate: {(total_classified/total_pharmacies)*100:.1f}%")
    print(f"\n📄 Summary saved to: {summary_file}")
    print(f"📁 All results saved to: {output_path}")

def main():
    import time
    parser = argparse.ArgumentParser(description="Process cached Apify data and run classification")
    parser.add_argument("--limit", type=int, help="Limit processing to first N files (for testing)")
    parser.add_argument("--output-dir", default="data/processed_50_state", help="Output directory")
    parser.add_argument("--skip-existing", action="store_true", help="Skip files that have already been processed")
    parser.add_argument("--batch-size", type=int, help="Process all cached files in batches of this size (implies --skip-existing)")
    parser.add_argument("--sleep", type=int, default=0, help="Seconds to sleep between batches (only with --batch-size)")
    parser.add_argument("--auto-batch", action="store_true", help="Automatically process all unprocessed files in batches (implies --skip-existing)")

    args = parser.parse_args()

    # Auto-batch mode: process all unprocessed files in batches
    if args.batch_size or args.auto_batch:
        batch_size = args.batch_size or 3
        sleep_time = args.sleep or 0
        print(f"🚀 Auto-batch mode: processing all unprocessed files in batches of {batch_size} (sleep {sleep_time}s between)")
        total_processed = 0
        while True:
            cached_data = load_cached_data(limit=batch_size, skip_existing=True, output_dir=args.output_dir)
            if not cached_data:
                print("✅ All cached files processed!")
                break
            print(f"📦 Batch: {len(cached_data)} locations")
            total_pharmacies = sum(len(pharmacies) for pharmacies in cached_data.values())
            print(f"🏥 Pharmacies in this batch: {total_pharmacies}")
            results = process_and_classify_data(cached_data, args.output_dir)
            total_processed += len(cached_data)
            print(f"Batch complete. Total locations processed so far: {total_processed}")
            if sleep_time > 0:
                print(f"Sleeping {sleep_time}s before next batch...")
                time.sleep(sleep_time)
        print("\n🎉 Automated batch processing complete!")
        return

    # Manual/single-batch mode
    if args.limit:
        print(f"🧪 Running trial with {args.limit} files...")
    else:
        print("🚀 Processing all cached data from 50-state run...")

    # Load cached data, skipping already processed if requested
    cached_data = load_cached_data(limit=args.limit, skip_existing=args.skip_existing, output_dir=args.output_dir)

    if not cached_data:
        print("❌ No cached data found!")
        return

    print(f"📦 Found data for {len(cached_data)} locations")
    total_pharmacies = sum(len(pharmacies) for pharmacies in cached_data.values())
    print(f"🏥 Total pharmacies to process: {total_pharmacies}")

    # Process and classify
    results = process_and_classify_data(cached_data, args.output_dir)

    print("\n✅ Processing complete!")

if __name__ == "__main__":
    main()
