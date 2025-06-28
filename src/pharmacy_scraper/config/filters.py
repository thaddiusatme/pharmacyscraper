"""Centralized configuration for pharmacy filtering rules.

This module contains comprehensive lists of terms and patterns used to identify
and filter out chain pharmacies, hospital-affiliated pharmacies, and other
unwanted results.
"""
from typing import Set, List, Dict, Any
import re

# Major pharmacy chains and retail stores with pharmacies
CHAIN_PHARMACIES: Set[str] = {
    # Major pharmacy chains
    "CVS", "CVS Pharmacy", "CVS/pharmacy", "CVS Health",
    "Walgreens", "Walgreens Pharmacy", "Walgreens Co",
    "Rite Aid", "RiteAid", "Rite-Aid",
    "Duane Reade", "Duane Reade Pharmacy",
    "Longs Drugs", "Longs Drugs Pharmacy",
    
    # Supermarkets with pharmacies
    "Kroger", "Kroger Pharmacy", "The Kroger Co",
    "Publix", "Publix Pharmacy", "Publix Super Markets",
    "Safeway", "Safeway Pharmacy",
    "Albertsons", "Albertsons Pharmacy",
    "Winn-Dixie", "Winn Dixie Pharmacy",
    "Giant Eagle", "Giant Eagle Pharmacy",
    "Meijer", "Meijer Pharmacy",
    "Hy-Vee", "Hy-Vee Pharmacy",
    "H-E-B", "H-E-B Pharmacy",
    "ShopRite", "ShopRite Pharmacy",
    "Wegmans", "Wegmans Pharmacy",
    "Stop & Shop", "Stop & Shop Pharmacy",
    "Harris Teeter", "Harris Teeter Pharmacy",
    "Food Lion", "Food Lion Pharmacy",
    "King Soopers", "King Soopers Pharmacy",
    "Fred Meyer", "Fred Meyer Pharmacy",
    "QFC", "QFC Pharmacy",
    "Ralphs", "Ralphs Pharmacy",
    
    # Wholesale clubs with pharmacies
    "Costco", "Costco Pharmacy", "Costco Wholesale",
    "Sam's Club", "Sams Club", "Sams Club Pharmacy",
    "BJ's", "BJ's Wholesale", "BJ's Pharmacy",
    
    # General merchandise stores with pharmacies
    "Walmart", "Walmart Pharmacy", "Walmart Inc",
    "Target", "Target Pharmacy", "Target Corporation",
    "Kmart", "Kmart Pharmacy",
    "Kohl's", "Kohl's Pharmacy",
    "Walgreens", "Walgreens Pharmacy", "Walgreens Boots Alliance"
}

# Hospital and healthcare system indicators
HOSPITAL_TERMS: Set[str] = {
    "Hospital", "Clinic", "Medical Center", "Health Center",
    "Health System", "Healthcare", "Medical Group", "Physicians",
    "VA", "Veterans Affairs", "Kaiser", "Kaiser Permanente",
    "Mayo Clinic", "Cleveland Clinic", "Johns Hopkins",
    "Cedars-Sinai", "Mass General", "Massachusetts General",
    "Brigham", "Women's Hospital", "Children's Hospital",
    "Memorial Hospital", "Regional Medical Center",
    "University Hospital", "Medical School", "HealthPartners",
    "ANMC", "Alaska Native Medical Center",
    "ANTHC", "Alaska Native Tribal Health Consortium"
}

# Additional exclusion terms (not hospitals or major chains)
EXCLUSION_TERMS: Set[str] = {
    "MinuteClinic", "The Little Clinic", "Take Care Clinic",
    "Urgent Care", "Emergency Room", "ER", "ICU", "Surgery",
    "Radiology", "Oncology", "Pediatrics", "Maternity", "Laboratory",
    "Phlebotomy", "Infusion Center", "Cancer Center", "Heart Institute",
    "Rehabilitation", "Physical Therapy", "Occupational Therapy",
    "Dialysis", "Pharmaceutical", "Research", "Clinical Trial",
    "Specialty Pharmacy", "Mail Order Pharmacy"
}

# Compiling regex patterns for more flexible matching
CHAIN_PATTERNS = [re.compile(rf'\b{re.escape(term.lower())}\b', re.IGNORECASE) 
                 for term in CHAIN_PHARMACIES]

HOSPITAL_PATTERNS = [re.compile(rf'\b{re.escape(term.lower())}\b', re.IGNORECASE) 
                    for term in HOSPITAL_TERMS]

EXCLUSION_PATTERNS = [re.compile(rf'\b{re.escape(term.lower())}\b', re.IGNORECASE) 
                     for term in EXCLUSION_TERMS]

def is_chain_pharmacy(name: str) -> bool:
    """Check if a pharmacy name matches known chain patterns."""
    if not name:
        return False
    name_lower = name.lower()
    return any(pattern.search(name_lower) for pattern in CHAIN_PATTERNS)

def is_hospital_pharmacy(name: str) -> bool:
    """Check if a pharmacy name is associated with a hospital."""
    if not name:
        return False
    name_lower = name.lower()
    return any(pattern.search(name_lower) for pattern in HOSPITAL_PATTERNS)

def should_exclude(name: str) -> bool:
    """Check if a result should be excluded based on name patterns."""
    if not name:
        return False
    name_lower = name.lower()
    return any(pattern.search(name_lower) for pattern in EXCLUSION_PATTERNS)

def filter_pharmacy(pharmacy: Dict[str, Any]) -> bool:
    """
    Determine if a pharmacy should be included in results.
    Returns True to keep, False to exclude.
    """
    name = pharmacy.get('name', '') or pharmacy.get('title', '')
    
    # Exclude chains and hospital pharmacies
    if is_chain_pharmacy(name) or is_hospital_pharmacy(name) or should_exclude(name):
        return False
        
    return True

# Export the complete set of all exclusion terms for reference
ALL_EXCLUSION_TERMS = CHAIN_PHARMACIES.union(HOSPITAL_TERMS).union(EXCLUSION_TERMS)
