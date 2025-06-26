#!/usr/bin/env python3
"""
Debug script to check if pandas objects are being passed to the classifier.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from dotenv import load_dotenv
from pharmacy_scraper.classification.perplexity_client import PerplexityClient
import logging

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def test_pandas_conversion():
    """Test to see what type of objects are being passed in the pipeline."""
    
    # Load the actual pipeline results to simulate the exact data
    df = pd.read_csv('data/pipeline_results/pharmacies_final.csv')
    
    # Get one failed pharmacy
    failed_pharmacy = df[df['method'] == 'api_failure'].iloc[0]
    
    print("=== TESTING PANDAS OBJECT TYPES ===")
    print(f"Failed pharmacy type: {type(failed_pharmacy)}")
    print(f"Failed pharmacy is pandas Series: {isinstance(failed_pharmacy, pd.Series)}")
    
    # Test what happens when we convert to dict
    pharmacy_dict = failed_pharmacy.to_dict()
    print(f"Converted to dict type: {type(pharmacy_dict)}")
    print(f"Sample keys: {list(pharmacy_dict.keys())[:5]}")
    
    # Test DataFrame.to_dict(orient="records") like in pipeline
    sample_df = df[df['method'] == 'api_failure'].head(1)
    records = sample_df.to_dict(orient="records")
    print(f"Records from DataFrame: {type(records)}")
    print(f"First record type: {type(records[0]) if records else 'No records'}")
    
    # Test what happens if we accidentally pass a Series
    try:
        print("=== TESTING SERIES ACCESS ===")
        # Simulate the error - try to access Series like a dict
        test_access = failed_pharmacy['title']  # This should work
        print(f"Series access works: {test_access}")
        
        # But this might cause issues in certain contexts
        print(f"Series .get() method: {hasattr(failed_pharmacy, 'get')}")
        
    except Exception as e:
        print(f"Error accessing Series: {e}")

if __name__ == "__main__":
    test_pandas_conversion()
