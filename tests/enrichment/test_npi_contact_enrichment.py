from types import SimpleNamespace
import pytest

from pharmacy_scraper.enrichment.npi_contact import enrich_contact_from_npi


def test_enrich_contact_from_npi_populates_authorized_official():
    """Test that NPI lookup populates contact fields from Authorized Official."""
    # Mock NPI data with Authorized Official
    npi_data = {
        "basic": {
            "authorized_official_last_name": "Smith",
            "authorized_official_first_name": "John",
            "authorized_official_middle_name": "A",
            "authorized_official_title_or_credential": "PharmD",
        }
    }
    
    result = enrich_contact_from_npi(npi_data)
    
    assert result["contact_name"] == "John A Smith"
    assert result["contact_role"] == "PharmD"
    assert result["contact_source"] == "npi_authorized_official"


def test_enrich_contact_from_npi_handles_minimal_data():
    """Test NPI enrichment with only first/last name."""
    npi_data = {
        "basic": {
            "authorized_official_last_name": "Johnson",
            "authorized_official_first_name": "Mary",
        }
    }
    
    result = enrich_contact_from_npi(npi_data)
    
    assert result["contact_name"] == "Mary Johnson"
    assert result["contact_role"] is None
    assert result["contact_source"] == "npi_authorized_official"


def test_enrich_contact_from_npi_returns_empty_when_no_authorized_official():
    """Test that empty NPI data returns empty contact fields."""
    npi_data = {"basic": {}}
    
    result = enrich_contact_from_npi(npi_data)
    
    assert result["contact_name"] is None
    assert result["contact_role"] is None
    assert result["contact_source"] is None


def test_enrich_contact_from_npi_handles_malformed_data():
    """Test that malformed/missing NPI data doesn't crash."""
    result = enrich_contact_from_npi(None)
    assert result["contact_name"] is None
    
    result = enrich_contact_from_npi({})
    assert result["contact_name"] is None
    
    result = enrich_contact_from_npi({"basic": None})
    assert result["contact_name"] is None


def test_enrich_contact_from_npi_does_not_overwrite_existing_fields():
    """Test that existing contact fields are preserved."""
    npi_data = {
        "basic": {
            "authorized_official_last_name": "Smith", 
            "authorized_official_first_name": "John",
        }
    }
    
    existing = {
        "contact_name": "Pre-existing Name",
        "contact_role": "Pre-existing Role",
        "contact_source": "api"
    }
    
    result = enrich_contact_from_npi(npi_data, existing_contact=existing)
    
    # Should preserve existing values
    assert result["contact_name"] == "Pre-existing Name"
    assert result["contact_role"] == "Pre-existing Role" 
    assert result["contact_source"] == "api"
