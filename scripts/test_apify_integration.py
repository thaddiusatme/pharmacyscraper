#!/usr/bin/env python3
"""
Test script for Apify integration with credit limits.

This script demonstrates how to use the ApifyPharmacyScraper with credit tracking
and rate limiting to safely test the integration without exceeding budget.

Usage:
    python scripts/test_apify_integration.py --state CA --city "San Francisco" --max-results 5
"""
import os
import argparse
import logging
import json
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

# Import the scraper
from src.dedup_self_heal.apify_integration import get_apify_client
from src.utils.api_usage_tracker import credit_tracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test Apify integration with credit limits')
    parser.add_argument('--state', type=str, required=True, help='Two-letter state code (e.g., CA)')
    parser.add_argument('--city', type=str, help='City name (optional)')
    parser.add_argument('--query', type=str, help='Custom search query (optional)')
    parser.add_argument('--max-results', type=int, default=5, help='Maximum results to fetch (default: 5)')
    parser.add_argument('--budget', type=float, help='Total API budget in credits (overrides env var)')
    parser.add_argument('--daily-limit', type=float, help='Daily API limit in credits (overrides env var)')
    parser.add_argument('--output', type=str, default='apify_results.json', help='Output file path (default: apify_results.json)')
    return parser.parse_args()

def main():
    """Main function to test Apify integration."""
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    args = parse_args()
    
    # Override budget if specified
    if args.budget is not None:
        credit_tracker.budget = args.budget
    if args.daily_limit is not None:
        credit_tracker.daily_limit = args.daily_limit
    
    # Print current usage
    usage = credit_tracker.get_usage_summary()
    print("\n=== API Credit Usage ===")
    print(f"Total budget: {usage['total_budget']:.2f} credits")
    print(f"Used: {usage['total_used']:.2f} credits")
    print(f"Remaining: {usage['remaining']:.2f} credits")
    print(f"Daily limit: {usage['daily_limit']:.2f} credits")
    print(f"Used today: {usage['today_used']:.2f} credits")
    
    # Check if we have enough credits
    if not credit_tracker.check_credit_available():
        print("\n❌ Not enough credits available. Please check your budget.")
        return
    
    # Initialize the scraper
    try:
        print(f"\n🔍 Searching for pharmacies in {args.city + ', ' if args.city else ''}{args.state}...")
        scraper = get_apify_client()
        
        # Run the search
        results = scraper.scrape_pharmacies(
            state=args.state,
            city=args.city,
            query=args.query,
            max_results=args.max_results
        )
        
        # Print results
        print(f"\n✅ Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['name']}")
            print(f"   📍 {result['address']}")
            if result.get('phone'):
                print(f"   📞 {result['phone']}")
            if result.get('website'):
                print(f"   🌐 {result['website']}")
        
        # Save results to file
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved to {output_path.absolute()}")
        
    except Exception as e:
        logger.error(f"Error during search: {str(e)}", exc_info=True)
    
    # Print final usage
    usage = credit_tracker.get_usage_summary()
    print("\n=== Final Credit Usage ===")
    print(f"Used this run: {usage['total_used'] - usage['total_used'] + (usage['today_used'] - usage['today_used']):.2f} credits")
    print(f"Total used: {usage['total_used']:.2f}/{usage['total_budget']:.2f} credits")
    print(f"Remaining: {usage['remaining']:.2f} credits")
    print(f"Used today: {usage['today_used']:.2f}/{usage['daily_limit']:.2f} credits")

if __name__ == "__main__":
    main()
