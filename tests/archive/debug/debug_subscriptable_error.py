#!/usr/bin/env python3
"""
Debug script to capture the exact stack trace of the subscriptable error.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from dotenv import load_dotenv
from pharmacy_scraper.classification.classifier import Classifier
from pharmacy_scraper.classification.perplexity_client import PerplexityClient
import logging
import traceback

# Load environment variables
load_dotenv()

# Set up detailed logging to capture everything
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')

def debug_exact_error():
    """Capture the exact stack trace of the subscriptable error."""
    
    # Get a pharmacy that will definitely cause the error (not cached)
    df = pd.read_csv('data/pipeline_results/pharmacies_final.csv')
    failed_pharmacy = df[df['method'] == 'api_failure'].iloc[1]  # Second failed pharmacy
    pharmacy_dict = failed_pharmacy.to_dict()
    
    print(f"=== DEBUGGING EXACT SUBSCRIPTABLE ERROR ===")
    print(f"Testing pharmacy: {pharmacy_dict.get('title', 'N/A')}")
    
    # Initialize client
    client = PerplexityClient(cache_dir="data/cache/classification")
    classifier = Classifier(client)
    
    # Call the exact method that fails, but with detailed exception handling
    try:
        result = classifier.classify_pharmacy(pharmacy_dict)
        print(f"Unexpected success: {result}")
    except Exception as e:
        print(f"\n💥 CAUGHT EXCEPTION: {type(e).__name__}: {e}")
        print(f"\n📍 FULL STACK TRACE:")
        traceback.print_exc()
        
        # Check if this is the subscriptable error
        if "not subscriptable" in str(e):
            print(f"\n🎯 CONFIRMED: This is the subscriptable error!")
            
            # Try to get more details about the error context
            tb = e.__traceback__
            print(f"\n📋 TRACEBACK ANALYSIS:")
            while tb:
                frame = tb.tb_frame
                code = frame.f_code
                print(f"  File: {code.co_filename}:{tb.tb_lineno}")
                print(f"  Function: {code.co_name}")
                print(f"  Local vars: {list(frame.f_locals.keys())}")
                tb = tb.tb_next

if __name__ == "__main__":
    debug_exact_error()
