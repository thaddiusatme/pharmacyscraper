#!/usr/bin/env python3
"""
Debug script to replicate exact pipeline classification conditions and capture errors.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from dotenv import load_dotenv
from pharmacy_scraper.classification.classifier import Classifier
from pharmacy_scraper.classification.perplexity_client import PerplexityClient
import logging

# Load environment variables
load_dotenv()

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def debug_classification_failures():
    """Replicate exact pipeline classification process to capture errors."""
    
    # Load pipeline results to get failed pharmacies
    df = pd.read_csv('data/pipeline_results/pharmacies_final.csv')
    
    # Get 3 failed pharmacies to test
    failed_pharmacies = df[df['method'] == 'api_failure'].head(3)
    
    print("=== REPLICATING PIPELINE CLASSIFICATION ===")
    
    # Convert to list of dicts exactly like pipeline does
    pharmacies_list = failed_pharmacies.to_dict(orient="records")
    
    print(f"Testing {len(pharmacies_list)} failed pharmacies...")
    print(f"First pharmacy keys: {list(pharmacies_list[0].keys())[:10]}...")
    
    # Initialize classifier exactly like pipeline does
    try:
        client = PerplexityClient(cache_dir="data/cache/classification")
        classifier = Classifier(client)
        print("✅ Classifier initialized successfully")
    except ValueError as e:
        print(f"❌ Failed to initialize PerplexityClient: {e}")
        return
    
    # Process each pharmacy exactly like pipeline loop
    for i, pharmacy in enumerate(pharmacies_list):
        print(f"\n=== TESTING PHARMACY {i+1}: {pharmacy.get('title', 'N/A')} ===")
        
        try:
            print(f"Pharmacy data type: {type(pharmacy)}")
            print(f"Is dict: {isinstance(pharmacy, dict)}")
            
            # This is the exact pipeline call
            result = classifier.classify_pharmacy(pharmacy)
            
            if result:
                print(f"✅ SUCCESS: {result.get('classification', 'unknown')} with confidence {result.get('confidence', 0)}")
            else:
                print("❌ FAILED: Returned None")
                
        except Exception as exc:
            print(f"💥 EXCEPTION: {type(exc).__name__}: {exc}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    debug_classification_failures()
