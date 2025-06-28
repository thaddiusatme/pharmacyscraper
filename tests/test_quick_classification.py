#!/usr/bin/env python3
"""
Quick test to verify Perplexity API is still working after the bug fix.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from pharmacy_scraper.classification.perplexity_client import PerplexityClient
import logging

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def test_new_pharmacy():
    """Test classification with a new pharmacy (not cached)."""
    
    # Different pharmacy data that shouldn't be cached
    pharmacy_data = {
        "title": "CVS Pharmacy #12345",
        "address": "123 Main St, Anytown, CA 90210",
        "categories": "Pharmacy",
        "website": "https://www.cvs.com"
    }
    
    try:
        client = PerplexityClient()
        print("📞 Testing classification of new pharmacy...")
        result = client.classify_pharmacy(pharmacy_data)
        
        if result:
            print(f"✅ Classification successful: {result}")
            return True
        else:
            print("❌ Classification failed (returned None)")
            return False
            
    except Exception as e:
        print(f"💥 Exception during classification: {e}")
        return False

if __name__ == "__main__":
    success = test_new_pharmacy()
    exit(0 if success else 1)
