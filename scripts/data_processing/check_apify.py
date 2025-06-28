#!/usr/bin/env python3
"""
Check Apify API token and list available actors.
"""
import os
import sys
from apify_client import ApifyClient

def main():
    # Get API token from environment
    api_token = os.getenv("APIFY_API_TOKEN") or os.getenv("APIFY_TOKEN")
    if not api_token:
        print("Error: APIFY_API_TOKEN or APIFY_TOKEN environment variable not set")
        sys.exit(1)

    # Initialize client
    client = ApifyClient(api_token)
    
    # Test token by getting user info
    try:
        user = client.user(None).get()
        print(f"✅ Successfully authenticated as: {user.get('username')} ({user.get('email')})")
        print(f"   Plan: {user.get('plan', {}).get('name', 'Unknown')}")
    except Exception as e:
        print(f"❌ Failed to authenticate with Apify: {e}")
        sys.exit(1)
    
    # List available actors (first page)
    print("\n🔍 Listing available actors (first 10):")
    try:
        actors = client.actors().list(limit=10).get('items', [])
        if not actors:
            print("No actors found. Your token might not have access to any actors.")
        else:
            for actor in actors:
                print(f"- {actor.get('username')}/{actor.get('name')} (v{actor.get('version')}): {actor.get('title')}")
    except Exception as e:
        print(f"❌ Failed to list actors: {e}")

if __name__ == "__main__":
    main()
