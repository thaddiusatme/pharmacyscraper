import json
import os
import sys
from datetime import datetime
import pandas as pd

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.classification.classifier import Classifier
from src.utils.logger import setup_logger

# --- Configuration ---
BACKFILL_CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.api_cache', 'apify'))
CLASSIFIED_OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'classified_backfill'))
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs'))

# --- Main Execution ---
def main():
    """Main function to process and classify backfill pharmacy data."""
    # --- Setup ---
    os.makedirs(CLASSIFIED_OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    log_file = os.path.join(LOG_DIR, f"process_backfill_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger = setup_logger('process_backfill_data', log_file)

    logger.info("--- Starting Backfill Data Classification ---")
    logger.info(f"Reading raw data from: {BACKFILL_CACHE_DIR}")
    logger.info(f"Saving classified data to: {CLASSIFIED_OUTPUT_DIR}")

    classifier = Classifier(force_reclassification=True)
    
    # --- Data Processing ---
    all_files = [f for f in os.listdir(BACKFILL_CACHE_DIR) if f.endswith('.json')]
    logger.info(f"Found {len(all_files)} JSON files to process.")

    state_counts = {
        'NH': {'total': 0, 'independent': 0},
        'VT': {'total': 0, 'independent': 0},
        'WV': {'total': 0, 'independent': 0},
        'WY': {'total': 0, 'independent': 0}
    }

    for filename in all_files:
        file_path = os.path.join(BACKFILL_CACHE_DIR, filename)
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Could not read or parse {filename}: {e}")
            continue

        if not data:
            logger.warning(f"No data in {filename}, skipping.")
            continue

        # Extract state from filename, assuming format like '..._in_city_st.json'
        try:
            state = filename.split('_')[-1].replace('.json', '').upper()
            if state not in state_counts:
                logger.warning(f"State '{state}' from filename '{filename}' not in target states. Skipping.")
                continue
        except IndexError:
            logger.error(f"Could not determine state from filename: {filename}")
            continue

        logger.info(f"Processing {len(data)} pharmacies from {filename} for state {state}.")
        state_counts[state]['total'] += len(data)

        pharmacies_df = pd.DataFrame(data)
        classified_pharmacies_df = classifier.classify_pharmacies(pharmacies_df)
        classified_pharmacies = classified_pharmacies_df.to_dict('records')

        # Filter for independents and count
        independent_pharmacies = [p for p in classified_pharmacies if p.get('classification') == 'independent']
        state_counts[state]['independent'] += len(independent_pharmacies)

        # --- Save Classified Data ---
        output_filename = os.path.join(CLASSIFIED_OUTPUT_DIR, f"classified_{filename}")
        with open(output_filename, 'w') as f:
            json.dump(classified_pharmacies, f, indent=4)
        logger.info(f"Saved {len(classified_pharmacies)} classified pharmacies to {output_filename}")

    # --- Reporting ---
    logger.info("--- Backfill Classification Summary ---")
    for state, counts in state_counts.items():
        logger.info(f"State: {state} | Total Processed: {counts['total']} | Independent Found: {counts['independent']}")
        if counts['independent'] < 25:
            logger.warning(f"{state} is still below the 25 independent pharmacy threshold.")
        else:
            logger.info(f"{state} now meets the threshold.")

    logger.info("--- Backfill Data Classification Finished ---")

if __name__ == "__main__":
    main()
