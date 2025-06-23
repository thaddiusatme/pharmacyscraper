import json
import os
import re
from collections import defaultdict
import argparse

def generate_final_report(data_dirs, report_path):
    """
    Scans specified data directories, aggregates pharmacy classification data,
    and generates a summary report.
    """
    # Regular expression to extract city and state from filenames
    # Handles formats like 'pharmacies_los_angeles_ca.json' and 'pharmacies_in_cheyenne_wy.json'
    filename_pattern = re.compile(r'pharmacies_(?:in_)?([a-z_-]+)_([a-z]{2})\.json')

    state_stats = defaultdict(lambda: defaultdict(int))
    overall_totals = defaultdict(int)

    all_files_to_process = []
    for data_dir in data_dirs:
        print(f"Scanning directory: {data_dir}")
        try:
            files_in_dir = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.json') and 'summary' not in f and 'filter' not in f]
            all_files_to_process.extend(files_in_dir)
        except FileNotFoundError:
            print(f"Warning: Directory not found, skipping: {data_dir}")
            continue

    if not all_files_to_process:
        print("No processed JSON files found to generate a report.")
        return

    for file_path in all_files_to_process:
        filename = os.path.basename(file_path)
        # The filenames from list_dir can sometimes have extra characters, so we'll clean them up
        clean_filename = filename.split('\t')[0].strip()

        match = filename_pattern.match(clean_filename)
        if not match:
            print(f"Warning: Skipping file with unexpected format: {clean_filename}")
            continue

        city, state = match.groups()
        state = state.upper()

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading or parsing {clean_filename}: {e}")
            continue

        # Aggregate stats from the file
        # The structure is a list of pharmacy dicts
        for pharmacy in data:
            classification = pharmacy.get('classification', 'unknown').lower()
            if classification == 'independent':
                state_stats[state]['independent'] += 1
            elif classification == 'chain':
                state_stats[state]['chain'] += 1
            elif classification == 'hospital':
                state_stats[state]['hospital'] += 1
            else:
                state_stats[state]['other'] += 1

            state_stats[state]['total_processed'] += 1

    # Calculate overall totals
    for state, counts in state_stats.items():
        overall_totals['independent'] += counts.get('independent', 0)
        overall_totals['chain'] += counts.get('chain', 0)
        overall_totals['hospital'] += counts.get('hospital', 0)
        overall_totals['other'] += counts.get('other', 0)
        overall_totals['total_processed'] += counts.get('total_processed', 0)

    # Generate the report content
    report_lines = []
    report_lines.append("=" * 50)
    report_lines.append("50-State Pharmacy Classification Summary Report")
    report_lines.append("=" * 50)
    report_lines.append("\n--- Overall Summary ---")
    report_lines.append(f"Total Pharmacies Processed: {overall_totals['total_processed']:,}")
    report_lines.append(f"  - Independent: {overall_totals['independent']:,}")
    report_lines.append(f"  - Chain:       {overall_totals['chain']:,}")
    report_lines.append(f"  - Hospital:    {overall_totals['hospital']:,}")
    if overall_totals['other'] > 0:
        report_lines.append(f"  - Other/Unknown: {overall_totals['other']:,}")
    report_lines.append("\n--- State-by-State Breakdown ---")

    # Sort states alphabetically
    sorted_states = sorted(state_stats.keys())

    for state in sorted_states:
        stats = state_stats[state]
        independent_count = stats.get('independent', 0)
        total_count = stats.get('total_processed', 0)
        
        warning = ""
        if independent_count < 25:
            warning = " (BELOW TARGET)"
            
        report_lines.append(f"\n{state}:{warning}")
        report_lines.append(f"  - Total Processed: {total_count:,}")
        report_lines.append(f"  - Independent:     {independent_count:,}")
        report_lines.append(f"  - Chain:           {stats.get('chain', 0):,}")
        report_lines.append(f"  - Hospital:        {stats.get('hospital', 0):,}")
        if stats.get('other', 0) > 0:
            report_lines.append(f"  - Other/Unknown:   {stats.get('other', 0):,}")

    report_content = "\n".join(report_lines)
    
    print(report_content)
    try:
        with open(report_path, 'w') as f:
            f.write(report_content)
        print(f"\n✅ Report successfully saved to {report_path}")
    except IOError as e:
        print(f"\n❌ Error saving report to file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a consolidated pharmacy classification report from multiple data directories.")
    parser.add_argument(
        '--data-dirs',
        nargs='+',
        required=True,
        help='One or more directories containing the processed JSON files.'
    )
    parser.add_argument(
        '--report-path',
        default='final_summary_report.txt',
        help='Path to save the final summary report.'
    )
    args = parser.parse_args()

    generate_final_report(args.data_dirs, args.report_path)
