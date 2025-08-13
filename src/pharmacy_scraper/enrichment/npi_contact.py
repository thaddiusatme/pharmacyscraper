from typing import Dict, Optional


def enrich_contact_from_npi(npi_data: Optional[Dict], existing_contact: Optional[Dict] = None) -> Dict[str, Optional[str]]:
    """
    Enrich contact fields from NPI Authorized Official data.
    
    Args:
        npi_data: NPI lookup result containing basic authorized official info
        existing_contact: Existing contact fields to preserve
        
    Returns:
        Dict with contact_name, contact_role, contact_source fields
    """
    # Initialize with None values
    result = {
        "contact_name": None,
        "contact_role": None,
        "contact_source": None
    }
    
    # If existing contact data is provided and populated, preserve it
    if existing_contact:
        for key in ["contact_name", "contact_role", "contact_source"]:
            if existing_contact.get(key) is not None:
                result[key] = existing_contact[key]
                
        # If all fields are already populated, return early
        if all(result[key] is not None for key in result):
            return result
    
    # Return early if no NPI data
    if not npi_data or not isinstance(npi_data, dict):
        return result
        
    # Extract basic info
    basic = npi_data.get("basic")
    if not basic or not isinstance(basic, dict):
        return result
        
    # Build contact name from Authorized Official fields
    first_name = basic.get("authorized_official_first_name", "").strip()
    middle_name = basic.get("authorized_official_middle_name", "").strip()
    last_name = basic.get("authorized_official_last_name", "").strip()
    
    if first_name or last_name:
        # Construct full name
        name_parts = [first_name]
        if middle_name:
            name_parts.append(middle_name)
        if last_name:
            name_parts.append(last_name)
            
        # Only set if we don't already have a contact_name
        if result["contact_name"] is None:
            result["contact_name"] = " ".join(name_parts)
            result["contact_source"] = "npi_authorized_official"
    
    # Extract contact role/credential
    credential = basic.get("authorized_official_title_or_credential", "").strip()
    if credential and result["contact_role"] is None:
        result["contact_role"] = credential
    
    return result
