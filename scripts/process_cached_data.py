CHAIN_PHARMACIES = {
    'cvs', 'walgreens', 'rite aid', 'walmart', 'target', 'costco', 'sams club',
    'kroger', 'publix', 'safeway', 'giant', 'stop & shop', 'wegmans', 'hannaford',
    'meijer', 'hy-vee', 'albertsons', 'vons', 'pavilions', 'harris teeter',
    'food lion', 'winn-dixie', 'heb', 'ralphs', 'fry\'s', 'smith\'s', 'fred meyer',
    'qfc', 'king soopers', 'duane reade', 'riteaid', 'wal-mart', 'wal mart',
    'walmart pharmacy', 'cvs pharmacy', 'walgreens pharmacy', 'rite aid pharmacy',
    'riteaid pharmacy', 'cvs/pharmacy', 'cvs store', 'walmart neighborhood market',
}

# Hospital/clinic/health system keywords (expand as needed)
HOSPITAL_KEYWORDS = {
    'hospital', 'clinic', 'medical center', 'va', 'veterans', 'kaiser', 'permanente',
    'health system', 'healthcare system', 'health care system', 'regional health',
    'university health', 'children\'s hospital', 'memorial hospital', 'health partners',
    'providence', 'sutter', 'banner', 'adventist', 'methodist', 'baptist health', 'st ',
    'saint ', 'ascension', 'mercy', 'integris', 'trinity', 'community health', 'clinic pharmacy',
    'ihs', 'indian health', 'tribal', 'native', 'anmc', 'anthc', 'army', 'navy', 'air force',
    'military', 'federal', 'government', 'county hospital', 'city hospital', 'university hospital',
    'academic medical', 'teaching hospital', 'veterans affairs', 'v.a.', 'v a ', 'va hospital',
}
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
        skipped_chains = 0
        skipped_hospitals = 0
        for i, pharmacy in enumerate(pharmacies):
            try:
                name = (pharmacy.get('title') or '').lower()
                # Pre-filter: skip known chains
                if any(chain in name for chain in CHAIN_PHARMACIES):
                    pharmacy_with_classification = pharmacy.copy()
                    pharmacy_with_classification['classification'] = 'chain'
                    pharmacy_with_classification['is_independent'] = False
                    pharmacy_with_classification['confidence'] = 1.0
                    pharmacy_with_classification['source'] = 'rule-based'
                    classified_pharmacies.append(pharmacy_with_classification)
                    skipped_chains += 1
                    continue
                # Pre-filter: skip known hospitals/clinics/health systems
                if any(hosp_kw in name for hosp_kw in HOSPITAL_KEYWORDS):
                    pharmacy_with_classification = pharmacy.copy()
                    pharmacy_with_classification['classification'] = 'hospital'
                    pharmacy_with_classification['is_independent'] = False
                    pharmacy_with_classification['confidence'] = 1.0
                    pharmacy_with_classification['source'] = 'rule-based'
                    classified_pharmacies.append(pharmacy_with_classification)
                    skipped_hospitals += 1
                    continue
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
        hospital_count = sum(1 for p in classified_pharmacies 
                         if p.get('classification') == 'hospital')
        error_count = sum(1 for p in classified_pharmacies 
                         if p.get('classification') in ['error', None])
        
        results_by_location[location] = {
            'total': len(classified_pharmacies),
            'independent': independent_count,
            'chain': chain_count,
            'hospital': hospital_count,
            'errors': error_count,
            'skipped_chains': skipped_chains,
            'skipped_hospitals': skipped_hospitals
        }
        
        print(f"    ✅ Saved to {location_file}")
        print(f"    📊 Independent: {independent_count}, Chain: {chain_count}, Hospital: {hospital_count}, Errors: {error_count}")
        print(f"    🚫 Skipped (pre-filtered): {skipped_chains} chains, {skipped_hospitals} hospitals/clinics")
    
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
    parser.add_argument("--skip-existing", action="store_true", help="Skip files that have already been processed (ignored if --states is used)")
    parser.add_argument("--batch-size", type=int, help="Process all cached files in batches of this size (implies --skip-existing)")
    parser.add_argument("--sleep", type=int, default=0, help="Seconds to sleep between batches (only with --batch-size)")
    parser.add_argument("--auto-batch", action="store_true", help="Automatically process all unprocessed files in batches (implies --skip-existing)")
    parser.add_argument("--states", type=str, help="Comma-separated list of state abbreviations (e.g. ca,wa,or) to process only those states. Forces a fresh run for those states.")

    args = parser.parse_args()

    # If --states is provided, filter cached files to only those states and force fresh classification
    if args.states:
        selected_states = set(s.strip().lower() for s in args.states.split(","))
        print(f"🔎 Limiting processing to states: {', '.join(selected_states).upper()}")
        # Load all cached data
        cached_data = load_cached_data(limit=None, skip_existing=False, output_dir=args.output_dir)
        # Filter locations by state
        filtered_data = {}
        for location, pharmacies in cached_data.items():
            # location is like 'los_angeles_ca' or 'portland_or'
            if location.endswith(tuple(f"_{state}" for state in selected_states)):
                filtered_data[location] = pharmacies
        if not filtered_data:
            print(f"❌ No cached data found for states: {', '.join(selected_states).upper()}")
            return
        print(f"📦 Found data for {len(filtered_data)} locations in selected states")
        total_pharmacies = sum(len(pharmacies) for pharmacies in filtered_data.values())
        print(f"🏥 Total pharmacies to process: {total_pharmacies}")
        results = process_and_classify_data(filtered_data, args.output_dir)
        print("\n✅ Processing complete!")
        return

    # Manual/single-batch mode (no --states)
    if args.limit:
        print(f"🧪 Running trial with {args.limit} files...")
    else:
        print("🚀 Processing all cached data from 50-state run...")

    cached_data = load_cached_data(limit=args.limit, skip_existing=args.skip_existing, output_dir=args.output_dir)

    if not cached_data:
        print("❌ No cached data found!")
        return

    print(f"📦 Found data for {len(cached_data)} locations")
    total_pharmacies = sum(len(pharmacies) for pharmacies in cached_data.values())
    print(f"🏥 Total pharmacies to process: {total_pharmacies}")

    results = process_and_classify_data(cached_data, args.output_dir)

    print("\n✅ Processing complete!")

if __name__ == "__main__":
    main()
