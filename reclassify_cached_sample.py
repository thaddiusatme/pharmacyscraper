#!/usr/bin/env python3
"""
Re-classify a sample of cached pharmacy data using our improved hospital pharmacy logic.
This will test the actual Perplexity API classification with our enhanced prompt.
"""

import json
import sys
import os
from pathlib import Path
import asyncio

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.classification.perplexity_client import PerplexityClient
from scripts.pharmacy_processor import CHAIN_PHARMACIES

# Sample files to test (starting with problematic ones)
SAMPLE_FILES = [
    "independent_pharmacy_anchorage_ak.json",  # Known hospital pharmacy issues
    "independent_pharmacy_fairbanks_ak.json", # Alaska - likely has VA/military pharmacies
    "independent_pharmacy_atlanta_ga.json",   # Large city - likely has hospital pharmacies
    "independent_pharmacy_chicago_il.json",   # Large city - likely has hospital pharmacies
    "independent_pharmacy_denver_co.json",    # Test case
]

def load_cached_file(filename):
    """Load a cached pharmacy data file."""
    cache_path = project_root / ".api_cache" / "apify" / filename
    
    if not cache_path.exists():
        print(f"❌ File not found: {filename}")
        return []
    
    with open(cache_path, 'r') as f:
        data = json.load(f)
    
    print(f"📥 Loaded {len(data)} pharmacies from {filename}")
    return data

def identify_potential_hospital_pharmacies(pharmacies):
    """Identify pharmacies that might be hospital-affiliated based on name patterns."""
    hospital_keywords = [
        'hospital', 'medical center', 'health system', 'va ', 'veterans',
        'anmc', 'anthc', 'clinic', 'medical', 'health', 'care center',
        'regional medical', 'memorial', 'general hospital', 'university hospital'
    ]
    
    potential_hospitals = []
    for pharmacy in pharmacies:
        name = pharmacy.get('title', '').lower()
        address = pharmacy.get('address', '').lower()
        
        if any(keyword in name or keyword in address for keyword in hospital_keywords):
            potential_hospitals.append(pharmacy)
    
    return potential_hospitals

async def reclassify_sample_files():
    """Re-classify sample cached files using improved logic."""
    
    print("🔄 RE-CLASSIFYING CACHED PHARMACY DATA WITH IMPROVED LOGIC")
    print("=" * 70)
    
    # Initialize classifier with our improved prompt
    classifier = PerplexityClient()
    
    total_stats = {
        'files_processed': 0,
        'pharmacies_processed': 0,
        'hospital_classifications': 0,
        'independent_classifications': 0,
        'chain_classifications': 0,
        'hospital_pharmacies_filtered': 0,
        'errors': 0
    }
    
    for filename in SAMPLE_FILES:
        print(f"\n📂 Processing: {filename}")
        print("-" * 50)
        
        # Load cached data
        pharmacies = load_cached_file(filename)
        if not pharmacies:
            continue
        
        # Focus on potential hospital pharmacies for efficiency
        potential_hospitals = identify_potential_hospital_pharmacies(pharmacies)
        
        if potential_hospitals:
            print(f"🏥 Found {len(potential_hospitals)} potential hospital pharmacies to re-classify")
        else:
            print(f"✅ No obvious hospital pharmacies found - testing first 3 pharmacies")
            potential_hospitals = pharmacies[:3]  # Test a few anyway
        
        results = []
        
        for i, pharmacy in enumerate(potential_hospitals):
            name = pharmacy.get('title', 'Unknown')
            address = pharmacy.get('address', 'Unknown')
            
            print(f"\n  {i+1}. Testing: {name}")
            print(f"     Address: {address}")
            
            # Check if already classified
            old_classification = pharmacy.get('classification', 'NONE')
            print(f"     Old: {old_classification}")
            
            try:
                # Re-classify with improved prompt
                print(f"     🔄 Calling Perplexity API...")
                
                # Create proper pharmacy data dictionary
                pharmacy_data = {
                    'title': name,
                    'name': name,
                    'address': address,
                    'location': address
                }
                
                classification_result = classifier.classify_pharmacy(pharmacy_data)
                
                # Debug: Show what we got back
                print(f"     📝 API returned type: {type(classification_result)}")
                print(f"     📝 API returned value: {str(classification_result)[:200]}...")
                
                # Handle both string and dict responses
                if isinstance(classification_result, str):
                    print(f"     ⚠️  Got string response, attempting to parse...")
                    # Parse the string response (likely JSON)
                    try:
                        import json
                        classification_result = json.loads(classification_result)
                        print(f"     ✅ Successfully parsed JSON")
                    except Exception as parse_error:
                        print(f"     ❌ JSON parsing failed: {parse_error}")
                        # If parsing fails, create a default dict
                        classification_result = {
                            'classification': 'unknown',
                            'confidence': 0.0,
                            'explanation': f'String response: {str(classification_result)[:50]}...'
                        }
                elif classification_result is None:
                    print(f"     ⚠️  Got None response from API")
                    classification_result = {
                        'classification': 'unknown',
                        'confidence': 0.0,
                        'explanation': 'API returned None'
                    }
                elif not isinstance(classification_result, dict):
                    print(f"     ⚠️  Got unexpected type: {type(classification_result)}")
                    classification_result = {
                        'classification': 'unknown',
                        'confidence': 0.0,
                        'explanation': f'Unexpected type: {type(classification_result)}'
                    }
                
                # Now safely extract values
                new_classification = classification_result.get('classification', 'unknown')
                confidence = classification_result.get('confidence', 0)
                explanation = classification_result.get('explanation', '')
                
                print(f"     New: {new_classification} (confidence: {confidence:.2f})")
                print(f"     Reason: {explanation}")
                
                # Update stats
                total_stats['pharmacies_processed'] += 1
                if new_classification == 'hospital':
                    total_stats['hospital_classifications'] += 1
                elif new_classification == 'independent':
                    total_stats['independent_classifications'] += 1
                elif new_classification == 'chain':
                    total_stats['chain_classifications'] += 1
                
                # Test post-processing filter (simple name-based check)
                pharm_name_lower = name.lower()
                was_filtered = any(chain in pharm_name_lower for chain in CHAIN_PHARMACIES)
                
                if was_filtered:
                    print(f"     Filter: 🚫 FILTERED OUT (matched: {[chain for chain in CHAIN_PHARMACIES if chain in pharm_name_lower]})")
                    if new_classification in ['hospital', 'chain']:
                        print(f"     Result: ✅ CORRECTLY FILTERED")
                    else:
                        print(f"     Result: ⚠️  UNEXPECTED FILTER (classified as {new_classification})")
                    total_stats['hospital_pharmacies_filtered'] += 1
                else:
                    print(f"     Filter: ✅ KEPT")
                
                # Show classification change
                if old_classification != 'NONE' and old_classification != new_classification:
                    print(f"     🔄 CLASSIFICATION CHANGED: {old_classification} → {new_classification}")
                
                results.append({
                    'name': name,
                    'old_classification': old_classification,
                    'new_classification': new_classification,
                    'confidence': confidence,
                    'filtered': was_filtered
                })
                
            except Exception as e:
                print(f"     ❌ Error: {e}")
                total_stats['errors'] += 1
                continue
        
        total_stats['files_processed'] += 1
        
        # Show file summary
        hospital_count = sum(1 for r in results if r['new_classification'] == 'hospital')
        independent_count = sum(1 for r in results if r['new_classification'] == 'independent')
        filtered_count = sum(1 for r in results if r['filtered'])
        
        print(f"\n  📊 {filename} Summary:")
        print(f"     🏥 Hospital: {hospital_count}")
        print(f"     🏪 Independent: {independent_count}")
        print(f"     🚫 Filtered out: {filtered_count}")
    
    # Overall summary
    print(f"\n📊 OVERALL SUMMARY")
    print("=" * 70)
    print(f"Files processed: {total_stats['files_processed']}")
    print(f"Pharmacies re-classified: {total_stats['pharmacies_processed']}")
    print(f"🏥 Hospital classifications: {total_stats['hospital_classifications']}")
    print(f"🏪 Independent classifications: {total_stats['independent_classifications']}")
    print(f"🔗 Chain classifications: {total_stats['chain_classifications']}")
    print(f"🚫 Pharmacies filtered out: {total_stats['hospital_pharmacies_filtered']}")
    print(f"❌ Errors: {total_stats['errors']}")
    
    if total_stats['hospital_classifications'] > 0:
        print(f"\n🎯 SUCCESS! Our improved logic is now correctly identifying hospital pharmacies!")
        print(f"   Ready to scale up to all {len(list((project_root / '.api_cache' / 'apify').glob('*.json')))} cached files.")
    else:
        print(f"\n⚠️  No hospital classifications found - this might indicate an issue or these files have no hospital pharmacies.")

if __name__ == "__main__":
    asyncio.run(reclassify_sample_files())
