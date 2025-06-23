CHAIN_PHARMACIES = {
    'cvs', 'walgreens', 'rite aid', 'walmart', 'costco', 'kinney', 'duane reade',
    'genoa', 'omnicare', 'star market', 'osco', 'medicine shoppe', 'market 32', "shaw's", 'shoprite',
    'kroger', 'publix', 'safeway', 'giant', 'stop & shop', 'wegmans', 'hannaford',
    'meijer', 'hy-vee', 'albertsons', 'vons', 'pavilions', 'harris teeter',
    'food lion', 'winn-dixie', 'heb', 'ralphs', 'fry\'s', 'smith\'s', 'fred meyer',
    'qfc', 'king soopers', 'duane reade', 'riteaid', 'wal-mart', 'wal mart',
    'walmart pharmacy', 'cvs pharmacy', 'walgreens pharmacy', 'rite aid pharmacy',
    'riteaid pharmacy', 'cvs/pharmacy', 'cvs store', 'walmart neighborhood market',
    'raley\'s', 'pharmerica', 'medicap', 'white drug', 'thrifty white',
    'baker\'s', 'macey\'s', 'sav-on', 'pharmaca', 'medly', 'pillpack',
    'longs drugs',  'times supermarket', 'price chopper', 'alto',
     'broulim\'s', 'harmons',
}

# Hospital/clinic/health system keywords (expand as needed)
HOSPITAL_KEYWORDS = {
    'hospital', 'clinic', 'medical center', 'care center', 'health', 'baystate', 'cha', 'wellness center',
    'kaiser', 'permanente',
    'health system', 'healthcare system', 'health care system', 'regional health',
    'university health', 'children\'s hospital', 'memorial hospital', 'health partners',
    'providence', 'sutter', 'banner', 'adventist', 'methodist', 'baptist health', 'st ',
    'integris', 'trinity', 'community health', 'clinic pharmacy',
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
from src.utils.logger import get_file_logger

def load_cached_data(limit: int = None, skip_existing: bool = False, output_dir: str = "data/processed_50_state", logger=None, input_dir: str = ".api_cache/apify") -> Dict[str, List[Dict]]:
    """Load cached pharmacy data from Apify, optionally limited to first N files"""
    if not logger:
        logger = logging.getLogger(__name__)

    cached_data = {}
    cache_path = Path(input_dir)
    output_path = Path(output_dir)
    logger.info(f"Scanning cache directory: {cache_path}")
    if not cache_path.exists():
        logger.warning(f"Cache directory {cache_path} does not exist")
        return cached_data
    json_files = list(cache_path.glob("*.json"))
    logger.info(f"Found {len(json_files)} cached files")

    # Filter out already processed files if skip_existing is set
    unprocessed_files = []
    for file_path in json_files:
        filename = file_path.stem
        if filename.startswith("independent_pharmacy_"):
            location = filename.replace("independent_pharmacy_", "")
            processed_file = output_path / f"pharmacies_{location}.json"
            if skip_existing and processed_file.exists():
                logger.info(f"Skipping already processed file: {processed_file}")
                continue
        unprocessed_files.append(file_path)

    # Apply limit after filtering
    if limit:
        unprocessed_files = unprocessed_files[:limit]
        logger.info(f"Processing first {len(unprocessed_files)} unprocessed files (limit={limit})")

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
                logger.info(f"Loaded {len(data)} pharmacies for {location}")
        except Exception as e:
            err_msg = f"Error loading {file_path}: {e}\n{traceback.format_exc()}"
            logger.error(err_msg)
    return cached_data

def process_and_classify_data(cached_data: Dict[str, List[Dict]], output_dir: str = "data/processed_50_state", logger=None, force_reclassification: bool = False):
    """Process cached data and run classification"""
    if not logger:
        logger = logging.getLogger(__name__)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize classifier
    classifier = Classifier(force_reclassification=force_reclassification)
    
    # Track statistics
    total_pharmacies = 0
    total_classified = 0
    results_by_location = {}
    all_api_classified = []

    logger.info("\n" + "="*60)
    logger.info("PROCESSING AND CLASSIFYING CACHED DATA")
    logger.info("="*60)
    
    for location, pharmacies in cached_data.items():
        logger.info(f"\nProcessing {location}: {len(pharmacies)} pharmacies")
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
                logger.info(f"  Classifying {i+1}/{len(pharmacies)}: {pharmacy_dict['name'][:50]}...")
                classification_result = classifier.classify_pharmacy(pharmacy_dict)
                # Add classification to pharmacy data
                pharmacy_with_classification = pharmacy.copy()
                pharmacy_with_classification.update(classification_result)
                pharmacy_with_classification['query_location'] = location # Add location for reporting
                classified_pharmacies.append(pharmacy_with_classification)
                all_api_classified.append(pharmacy_with_classification)
                total_classified += 1
            except Exception as e:
                err_msg = f"Failed to classify {pharmacy.get('title', '')}: {e}\n{traceback.format_exc()}"
                logger.error(err_msg)
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
        
        logger.info(f"    ✅ Saved to {location_file}")
        logger.info(f"    📊 Independent: {independent_count}, Chain: {chain_count}, Hospital: {hospital_count}, Errors: {error_count}")
        logger.info(f"    🚫 Skipped (pre-filtered): {skipped_chains} chains, {skipped_hospitals} hospitals/clinics")
    
    # Create summary report
    create_summary_report(results_by_location, output_path, total_pharmacies, total_classified)
    
    return results_by_location, all_api_classified

def create_summary_report(results_by_location: Dict, output_path: Path, total_pharmacies: int, total_classified: int, logger=None):
    """Create a summary report of the processing results"""
    if not logger:
        logger = logging.getLogger(__name__)
    
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
    logger.info("\n" + "="*60)
    logger.info("PROCESSING SUMMARY")
    logger.info("="*60)
    logger.info(f"📍 Locations processed: {len(results_by_location)}")
    logger.info(f"🏥 Total pharmacies: {total_pharmacies}")
    logger.info(f"✅ Successfully classified: {total_classified}")
    logger.info(f"📊 Independent: {total_independent}")
    logger.info(f"🏪 Chain: {total_chain}")
    logger.info(f"❌ Errors: {total_errors}")
    logger.info(f"💯 Success rate: {(total_classified/total_pharmacies)*100:.1f}%")
    logger.info(f"\n📄 Summary saved to: {summary_file}")
    logger.info(f"📁 All results saved to: {output_path}")

def check_state_pharmacy_counts(results_by_location: Dict, threshold: int = 25, logger=None):
    """Analyzes results to check if each state meets the minimum pharmacy threshold."""
    if not logger:
        logger = logging.getLogger(__name__)

    state_summary = {}
    # location is like 'los_angeles_ca' or 'portland_or'
    for location, stats in results_by_location.items():
        state = location.split('_')[-1]
        if len(state) == 2: # Basic check for a state abbreviation
            if state not in state_summary:
                state_summary[state] = {'independent': 0, 'locations': 0}
            state_summary[state]['independent'] += stats.get('independent', 0)
            state_summary[state]['locations'] += 1

    if not state_summary:
        logger.info("No state-level results to check.")
        return

    logger.info("\n" + "="*60)
    logger.info("STATE-LEVEL INDEPENDENT PHARMACY COUNT CHECK")
    logger.info(f"Target: {threshold} independent pharmacies per state")
    logger.info("="*60)
    
    all_states_ok = True
    for state, data in sorted(state_summary.items()):
        count = data['independent']
        if count < threshold:
            logger.warning(
                f"⚠️  State '{state.upper()}' has only {count} independent pharmacies across {data['locations']} locations. (BELOW TARGET)"
            )
            all_states_ok = False
        else:
            logger.info(
                f"✅ State '{state.upper()}' has {count} independent pharmacies across {data['locations']} locations. (OK)"
            )
    
    if all_states_ok:
        logger.info("\n👍 All processed states meet or exceed the minimum threshold.")

def main():
    import time
    logger = get_file_logger('process_cached_data')
    parser = argparse.ArgumentParser(description="Process cached Apify data and run classification")
    parser.add_argument("--limit", type=int, help="Limit processing to first N files (for testing)")
    parser.add_argument("--output-dir", default="data/processed_50_state", help="Output directory")
    parser.add_argument("--skip-existing", action="store_true", help="Skip files that have already been processed (ignored if --states is used)")
    parser.add_argument("--batch-size", type=int, help="Process all cached files in batches of this size (implies --skip-existing)")
    parser.add_argument("--sleep", type=int, default=0, help="Seconds to sleep between batches (only with --batch-size)")
    parser.add_argument("--auto-batch", action="store_true", help="Automatically process all unprocessed files in batches (implies --skip-existing)")
    parser.add_argument("--states", type=str, help="Comma-separated list of state abbreviations (e.g. ca,wa,or) to process only those states. Forces a fresh run for those states.")
    parser.add_argument("--input-dir", type=str, default=".api_cache/apify", help="Directory containing the cached Apify data files.")
    parser.add_argument("--force-reclassification", action="store_true", help="Force re-classification of all items, ignoring any cached results.")

    args = parser.parse_args()

    # If --states is provided, filter cached files to only those states and force fresh classification
    if args.states:
        selected_states = set(s.strip().lower() for s in args.states.split(","))
        logger.info(f"🔎 Limiting processing to states: {', '.join(selected_states).upper()}")
        # Load all cached data
        cached_data = load_cached_data(limit=None, skip_existing=False, output_dir=args.output_dir, logger=logger, input_dir=args.input_dir)
        # Filter locations by state
        filtered_data = {}
        for location, pharmacies in cached_data.items():
            # location is like 'los_angeles_ca' or 'portland_or'
            if location.endswith(tuple(f"_{state}" for state in selected_states)):
                filtered_data[location] = pharmacies
        if not filtered_data:
            logger.warning(f"❌ No cached data found for states: {', '.join(selected_states).upper()}")
            return
        logger.info(f"📦 Found data for {len(filtered_data)} locations in selected states")
        total_pharmacies = sum(len(pharmacies) for pharmacies in filtered_data.values())
        logger.info(f"🏥 Total pharmacies to process: {total_pharmacies}")
        results, classified_by_api = process_and_classify_data(filtered_data, args.output_dir, logger=logger, force_reclassification=args.force_reclassification)
        check_state_pharmacy_counts(results, logger=logger)
        create_filter_improvement_report(classified_by_api, Path(args.output_dir), logger=logger)
        logger.info("\n✅ Processing complete!")
        return

    # Manual/single-batch mode (no --states)
    if args.limit:
        logger.info(f"🧪 Running trial with {args.limit} files...")
    else:
        logger.info("🚀 Processing all cached data from 50-state run...")

    cached_data = load_cached_data(limit=args.limit, skip_existing=args.skip_existing, output_dir=args.output_dir, logger=logger, input_dir=args.input_dir)

    if not cached_data:
        logger.warning("❌ No cached data found!")
        return

    logger.info(f"📦 Found data for {len(cached_data)} locations")
    total_pharmacies = sum(len(pharmacies) for pharmacies in cached_data.values())
    logger.info(f"🏥 Total pharmacies to process: {total_pharmacies}")

    results, classified_by_api = process_and_classify_data(cached_data, args.output_dir, logger=logger, force_reclassification=args.force_reclassification)

    check_state_pharmacy_counts(results, logger=logger)
    create_filter_improvement_report(classified_by_api, Path(args.output_dir), logger=logger)
    logger.info("\n✅ Processing complete!")


def create_filter_improvement_report(classified_by_api: List[Dict], output_path: Path, logger=None):
    """Creates a CSV report of chains/hospitals that were classified by the API."""
    if not logger:
        logger = logging.getLogger(__name__)

    report_file = output_path / "filter_improvement_suggestions.csv"
    
    # Filter for chains/hospitals classified by the 'perplexity' source
    suggestions = [
        p for p in classified_by_api 
        if p.get('source') == 'perplexity' and p.get('classification') in ['chain', 'hospital']
    ]

    if not suggestions:
        logger.info("No new chain/hospital classifications from API to suggest for filtering.")
        return

    # Create a DataFrame for easy CSV export
    df_data = []
    for p in suggestions:
        df_data.append({
            'name': p.get('title'),
            'classification': p.get('classification'),
            'address': p.get('address'),
            'city_state': p.get('query_location'),
        })

    df = pd.DataFrame(df_data)
    df.to_csv(report_file, index=False)

    logger.info("\n" + "="*60)
    logger.info("FILTER IMPROVEMENT REPORT")
    logger.info("="*60)
    logger.info(f"Found {len(suggestions)} pharmacies classified as chain/hospital by the API.")
    logger.info(f"These could potentially be pre-filtered in the future.")
    logger.info(f"📝 Report saved to: {report_file}")

if __name__ == "__main__":
    main()
