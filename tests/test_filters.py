"""Tests for the centralized pharmacy filters."""
import pytest
from pharmacy_scraper.config.filters import (
    is_chain_pharmacy,
    is_hospital_pharmacy,
    should_exclude,
    filter_pharmacy,
    CHAIN_PHARMACIES,
    HOSPITAL_TERMS,
    EXCLUSION_TERMS
)

# Test data
TEST_PHARMACIES = [
    # Independent pharmacies (should pass)
    {"name": "Sunrise Pharmacy", "title": "Sunrise Pharmacy"},
    {"name": "Downtown Drug Store", "title": "Downtown Drug Store"},
    {"name": "Main Street Compounding", "title": "Main Street Compounding"},
    
    # Chain pharmacies (should be filtered out)
    {"name": "CVS Pharmacy #123", "title": "CVS Pharmacy #123"},
    {"name": "Walgreens", "title": "Walgreens"},
    {"name": "Rite Aid", "title": "Rite Aid"},
    
    # Hospital pharmacies (should be filtered out)
    {"name": "Mayo Clinic Pharmacy", "title": "Mayo Clinic Pharmacy"},
    {"name": "Phoenix VA Pharmacy", "title": "Phoenix VA Pharmacy"},
    {"name": "Children's Hospital Pharmacy", "title": "Children's Hospital Pharmacy"},
    
    # Other exclusions (should be filtered out)
    {"name": "MinuteClinic", "title": "MinuteClinic"},
    {"name": "Urgent Care Pharmacy", "title": "Urgent Care Pharmacy"},
    {"name": "Cancer Center Pharmacy", "title": "Cancer Center Pharmacy"}
]

# Test chain pharmacy detection
@pytest.mark.parametrize("name,expected", [
    ("CVS Pharmacy", True),
    ("Walgreens #123", True),
    ("Rite Aid Pharmacy", True),
    ("Local Drug Store", False),
    ("", False),
    (None, False)
])
def test_is_chain_pharmacy(name, expected):
    """Test chain pharmacy detection."""
    assert is_chain_pharmacy(name) == expected

# Test hospital pharmacy detection
@pytest.mark.parametrize("name,expected", [
    ("Mayo Clinic Pharmacy", True),
    ("Phoenix VA Medical Center Pharmacy", True),
    ("Children's Hospital Pharmacy", True),
    ("Local Drug Store", False),
    ("", False),
    (None, False)
])
def test_is_hospital_pharmacy(name, expected):
    """Test hospital pharmacy detection."""
    assert is_hospital_pharmacy(name) == expected

# Test exclusion terms
@pytest.mark.parametrize("name,expected", [
    ("MinuteClinic", True),
    ("Urgent Care Pharmacy", True),
    ("Cancer Center Pharmacy", True),
    ("Local Drug Store", False),
    ("", False),
    (None, False)
])
def test_should_exclude(name, expected):
    """Test exclusion term detection."""
    assert should_exclude(name) == expected

# Test full pharmacy filtering
def test_filter_pharmacy():
    """Test complete pharmacy filtering."""
    filtered = [p for p in TEST_PHARMACIES if filter_pharmacy(p)]
    
    # Should only keep the first 3 independent pharmacies
    assert len(filtered) == 3
    assert all(p["name"] in ["Sunrise Pharmacy", "Downtown Drug Store", "Main Street Compounding"] 
               for p in filtered)

# Test filter sets are not empty
def test_filter_sets_not_empty():
    """Ensure filter sets are not empty."""
    assert len(CHAIN_PHARMACIES) > 0
    assert len(HOSPITAL_TERMS) > 0
    assert len(EXCLUSION_TERMS) > 0

# Test case sensitivity
def test_case_insensitivity():
    """Test that filtering is case-insensitive."""
    assert is_chain_pharmacy("cvs pharmacy") is True
    assert is_chain_pharmacy("CVS PHARMACY") is True
    assert is_hospital_pharmacy("mayo clinic") is True
    assert is_hospital_pharmacy("MAYO CLINIC") is True
    assert should_exclude("minuteclinic") is True
    assert should_exclude("MINUTECLINIC") is True
