#!/usr/bin/env python3
"""
Debug script to check environment variable loading.
"""
import os
from dotenv import load_dotenv

print("=== DEBUGGING ENVIRONMENT VARIABLE LOADING ===")

# Test without loading .env
print(f"Before load_dotenv(): PERPLEXITY_API_KEY = {os.getenv('PERPLEXITY_API_KEY', 'NOT_FOUND')}")

# Load .env
load_dotenv()
print(f"After load_dotenv(): PERPLEXITY_API_KEY = {os.getenv('PERPLEXITY_API_KEY', 'NOT_FOUND')}")

# Check if it's actually a string and has content
api_key = os.getenv('PERPLEXITY_API_KEY')
if api_key:
    print(f"API key type: {type(api_key)}")
    print(f"API key length: {len(api_key)}")
    print(f"API key prefix: {api_key[:15]}...")
    print(f"API key is valid string: {isinstance(api_key, str) and len(api_key) > 10}")
else:
    print("❌ API key is None or empty")

# Test the exact same pattern as PerplexityClient
test_key = api_key or os.getenv("PERPLEXITY_API_KEY")
print(f"Using PerplexityClient pattern: {test_key[:15] if test_key else 'NO KEY'}...")
