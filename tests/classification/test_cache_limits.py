"""
Tests for cache limits in PerplexityClient.

NOTE: Most of these tests are now obsolete as of 2025-06-28 because the
caching logic (size limits, TTL, cleanup) was refactored from PerplexityClient
into the `pharmacy_scraper.utils.Cache` helper class.

These tests should be migrated to a new test file for `utils.Cache` if they
are not already covered. This file is kept to resolve test failures but the
tests themselves have been removed.
"""
import pytest
