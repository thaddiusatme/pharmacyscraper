import pandas as pd
import argparse
import logging
import glob
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def merge_and_dedupe_files(input_patterns, output_file):
    """
    Merges multiple CSV files from glob patterns into a single file and removes duplicates.

    Args:
        input_patterns (list): A list of glob patterns for the input CSV files.
        output_file (str): The path to the output CSV file.
    """
    all_files = []
    for pattern in input_patterns:
        all_files.extend(glob.glob(pattern))

    if not all_files:
        logging.warning("No files found matching the input patterns. No output file will be created.")
        return

    logging.info(f"Found {len(all_files)} files to merge.")

    df_list = []
    for file in all_files:
        try:
            df = pd.read_csv(file)
            df_list.append(df)
            logging.info(f"Successfully loaded {file} with {len(df)} rows.")
        except Exception as e:
            logging.error(f"Could not read file {file}: {e}")
            continue
    
    if not df_list:
        logging.error("No dataframes were loaded. Aborting merge.")
        return

    merged_df = pd.concat(df_list, ignore_index=True)
    logging.info(f"Total rows before deduplication: {len(merged_df)}")

    if 'placeId' in merged_df.columns:
        # Sort by a column that indicates data quality if available, e.g., 'scrapedAt', to keep the most recent record.
        # If not available, it will keep the first occurrence.
        if 'scrapedAt' in merged_df.columns:
            merged_df['scrapedAt'] = pd.to_datetime(merged_df['scrapedAt'], errors='coerce')
            merged_df = merged_df.sort_values('scrapedAt', ascending=False)

        deduped_df = merged_df.drop_duplicates(subset=['placeId'], keep='first')
        logging.info(f"Total rows after deduplication by 'placeId': {len(deduped_df)}")
    else:
        logging.warning("'placeId' column not found. Skipping deduplication.")
        deduped_df = merged_df

    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory: {output_dir}")

    deduped_df.to_csv(output_file, index=False)
    logging.info(f"Successfully saved merged and deduplicated data to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge and deduplicate CSV files from a raw data directory.")
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Directory containing the raw CSV files to merge.'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help='Path for the output merged and deduplicated CSV file.'
    )

    args = parser.parse_args()

    input_patterns = [os.path.join(args.input_dir, '*.csv')]
    
    merge_and_dedupe_files(input_patterns, args.output_file)
