
#!/usr/bin/env python3
"""
Analyze trial run data and compute metrics.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Set
import pandas as pd
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/trial_analysis.log')
    ]
)
logger = logging.getLogger(__name__)

class TrialAnalyzer:
    def __init__(self, data_dir: str):
        """Initialize with the directory containing trial data."""
        self.data_dir = Path(data_dir)
        self.pharmacies: List[Dict[str, Any]] = []
        self.metrics: Dict[str, Any] = defaultdict(int)
        self.unique_pharmacies: Set[str] = set()
        self.chain_names: Set[str] = set()
        
    def load_data(self) -> None:
        """Load all JSON files from the trial directory."""
        logger.info(f"Loading data from {self.data_dir}")
        
        for json_file in self.data_dir.rglob('*.json'):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.pharmacies.extend(data)
                        logger.info(f"Loaded {len(data)} pharmacies from {json_file}")
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")
    
    def compute_metrics(self) -> None:
        """Compute metrics from the loaded data."""
        if not self.pharmacies:
            logger.warning("No pharmacy data loaded")
            return
        
        # Basic counts
        self.metrics['total_pharmacies'] = len(self.pharmacies)
        
        # Track unique pharmacies by place_id and name+address
        unique_by_id = set()
        unique_by_name_addr = set()
        
        for pharm in self.pharmacies:
            # Unique by place_id
            if 'placeId' in pharm:
                unique_by_id.add(pharm['placeId'])
            
            # Unique by name + address
            name_addr = f"{pharm.get('title', '')}::{pharm.get('address', '')}"
            unique_by_name_addr.add(name_addr)
            
            # Track chain names (non-independent)
            if 'title' in pharm and 'independent' not in pharm['title'].lower():
                self.chain_names.add(pharm['title'])
        
        self.metrics['unique_by_place_id'] = len(unique_by_id)
        self.metrics['unique_by_name_addr'] = len(unique_by_name_addr)
        self.metrics['duplicate_rate'] = (
            1 - (len(unique_by_name_addr) / len(self.pharmacies))
            if self.pharmacies else 0
        )
        
        # Field completeness
        fields = ['title', 'address', 'phone', 'website', 'placeId']
        field_counts = {field: 0 for field in fields}
        
        for pharm in self.pharmacies:
            for field in fields:
                if field in pharm and pharm[field]:
                    field_counts[field] += 1
        
        self.metrics['field_completeness'] = {
            field: count / len(self.pharmacies)
            for field, count in field_counts.items()
        }
        
        # Chain vs independent
        self.metrics['chain_pharmacies'] = len(self.chain_names)
        self.metrics['independent_pharmacies'] = len(self.pharmacies) - len(self.chain_names)
    
    def generate_report(self) -> str:
        """Generate a markdown report of the metrics."""
        if not self.metrics:
            return "No metrics computed. Run compute_metrics() first."
        
        report = [
            "# Trial Run Analysis Report",
            f"**Data Directory:** {self.data_dir}",
            f"**Total Pharmacies Collected:** {self.metrics['total_pharmacies']}",
            f"**Unique Pharmacies (by place_id):** {self.metrics['unique_by_place_id']}",
            f"**Unique Pharmacies (by name+address):** {self.metrics['unique_by_name_addr']}",
            f"**Duplicate Rate:** {self.metrics['duplicate_rate']:.2%}",
            "\n## Field Completeness",
        ]
        
        for field, completeness in self.metrics['field_completeness'].items():
            report.append(f"- **{field}:** {completeness:.2%}")
        
        report.extend([
            "\n## Pharmacy Types",
            f"**Independent Pharmacies:** {self.metrics['independent_pharmacies']}",
            f"**Chain Pharmacies:** {self.metrics['chain_pharmacies']}",
        ])
        
        if self.chain_names:
            report.append("\n## Detected Chain Names")
            for name in sorted(self.chain_names):
                report.append(f"- {name}")
        
        return "\n".join(report)
    
    def save_report(self, output_path: str) -> None:
        """Save the report to a file."""
        report = self.generate_report()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"Report saved to {output_path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze trial run data')
    parser.add_argument('--data-dir', default='data/raw/trial_20240619',
                       help='Directory containing trial data')
    parser.add_argument('--output', default='reports/trial_analysis_20240619.md',
                       help='Output file for the analysis report')
    
    args = parser.parse_args()
    
    analyzer = TrialAnalyzer(args.data_dir)
    analyzer.load_data()
    analyzer.compute_metrics()
    
    # Print to console
    print("\n" + "="*50)
    print(analyzer.generate_report())
    print("="*50 + "\n")
    
    # Save to file
    analyzer.save_report(args.output)

if __name__ == "__main__":
    main()