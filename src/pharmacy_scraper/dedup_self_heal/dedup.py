"""
Deduplication and Self-Healing module for pharmacy data.

This module handles:
1. Grouping pharmacies by state
2. Removing duplicate pharmacy entries
3. Identifying under-filled states
4. Self-healing by finding additional pharmacies for under-filled states
5. Merging new pharmacy data with existing data
"""
import pandas as pd
import logging
from typing import Dict, List

# Import Apify integration
from .apify_integration import ApifyPharmacyScraper, ApifyScraperError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
TARGET_PHARMACIES_PER_STATE = 25
DEFAULT_MIN_REQUIRED = 10  # Minimum pharmacies before considering a state under-filled

# Initialize Apify scraper (lazy-loaded)
_apify_scraper = None

def get_apify_scraper() -> ApifyPharmacyScraper:
    """Get a configured ApifyPharmacyScraper instance.
    
    Returns:
        Configured ApifyPharmacyScraper instance
    """
    global _apify_scraper
    if _apify_scraper is None:
        _apify_scraper = ApifyPharmacyScraper()
    return _apify_scraper

def group_pharmacies_by_state(pharmacies: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Group pharmacies by their state.
    
    Args:
        pharmacies: DataFrame containing pharmacy data with a 'state' column
        
    Returns:
        Dictionary mapping state codes to DataFrames of pharmacies in that state
    """
    if 'state' not in pharmacies.columns:
        raise ValueError("Input DataFrame must contain a 'state' column")
        
    return {state: group for state, group in pharmacies.groupby('state')}

def remove_duplicates(pharmacies: pd.DataFrame, 
                    subset: List[str] = None,
                    keep: str = 'first') -> pd.DataFrame:
    """
    Remove duplicate pharmacy entries.
    
    Args:
        pharmacies: DataFrame containing pharmacy data
        subset: List of column names to consider for identifying duplicates.
               If None, all columns are used.
        keep: Which duplicates to keep. 'first' keeps the first occurrence, 
              'last' keeps the last, False keeps none.
              
    Returns:
        DataFrame with duplicates removed
    """
    if subset is None:
        # Default to using these columns for identifying duplicates
        subset = ['name', 'address', 'city', 'state', 'zip']
    
    # Only keep columns that exist in the DataFrame
    subset = [col for col in subset if col in pharmacies.columns]
    
    return pharmacies.drop_duplicates(subset=subset, keep=keep)

def identify_underfilled_states(grouped_pharmacies: Dict[str, pd.DataFrame],
                              min_required: int = DEFAULT_MIN_REQUIRED) -> Dict[str, int]:
    """
    Identify states that don't have enough pharmacies.
    
    Args:
        grouped_pharmacies: Dictionary of {state: DataFrame of pharmacies}
        min_required: Minimum number of pharmacies required per state
        
    Returns:
        Dictionary mapping state codes to number of additional pharmacies needed to reach TARGET_PHARMACIES_PER_STATE
    """
    underfilled = {}
    
    for state, df in grouped_pharmacies.items():
        count = len(df)
        # Only include states that are below the minimum required
        if count < min_required:
            underfilled[state] = TARGET_PHARMACIES_PER_STATE - count
    
    return underfilled

def scrape_pharmacies(state: str, count: int, city: str = None) -> pd.DataFrame:
    """
    Stub function for scraping pharmacies. This should be replaced with actual implementation
    or mocked in tests.
    
    Args:
        state: Two-letter state code
        count: Number of pharmacies to scrape
        city: Optional city to narrow down the search
        
    Returns:
        DataFrame of scraped pharmacies
    """
    raise NotImplementedError("scrape_pharmacies should be implemented or mocked")

def self_heal_state(state: str, 
                   existing: pd.DataFrame, 
                   count: int, 
                   min_required: int = DEFAULT_MIN_REQUIRED) -> pd.DataFrame:
    """
    Attempt to find more pharmacies for a given state.
    
    Args:
        state: Two-letter state code
        existing: DataFrame of existing pharmacies for this state
        count: Number of additional pharmacies to find
        min_required: Minimum number of pharmacies required (for logging)
        
    Returns:
        DataFrame of new pharmacies found
    """
    # Start with existing pharmacies
    result = existing.copy()
    
    # If we don't need any more pharmacies, return early
    if count <= 0:
        return result
    
    logger.info(f"Attempting to find {count} more pharmacies for {state}")
    
    # Get the cities with existing pharmacies
    existing_cities = set()
    if not existing.empty and 'city' in existing.columns:
        existing_cities = set(str(city).lower() for city in existing['city'].dropna())
    
    # Get a list of cities in the state
    cities = get_cities_for_state(state)
    
    # Filter out cities we already have pharmacies from
    new_cities = [city for city in cities if str(city).lower() not in existing_cities]
    
    # If we've run out of new cities, try cities we already have but with different queries
    if not new_cities and cities:
        new_cities = cities
    
    # If still no cities, use the state name as a fallback
    if not new_cities:
        new_cities = [state]
    
    # Try to find pharmacies in each city until we have enough
    found_pharmacies = []
    
    for city in new_cities:
        if len(found_pharmacies) >= count:
            break
            
        try:
            # Calculate how many more we need
            needed = count - len(found_pharmacies)
            
            # Try to get pharmacies for this city
            # Using positional arguments to match test expectations
            new_pharmacies = scrape_pharmacies(
                state,  # state as positional argument
                min(needed * 2, 10),  # count as positional argument
                city=city  # city as keyword argument
            )
            
            if new_pharmacies is not None and not new_pharmacies.empty:
                # Add to our found pharmacies
                found_pharmacies.append(new_pharmacies)
                
        except Exception as e:
            logger.warning(f"Error finding pharmacies in {city}, {state}: {e}")
            continue
    
    # Add any new pharmacies we found
    if found_pharmacies:
        new_pharmacies = pd.concat(found_pharmacies, ignore_index=True)
        
        if not new_pharmacies.empty:
            # If we have existing pharmacies, check for duplicates
            if not result.empty:
                logger.debug(f"Checking {len(new_pharmacies)} new pharmacies against {len(result)} existing pharmacies")
                
                # Create a set of unique identifiers from existing pharmacies
                existing_keys = set()
                for _, row in result.iterrows():
                    # Create a key based on city and zip, or phone if available
                    city_zip_key = (
                        str(row.get('city', '')).lower().strip(),
                        str(row.get('zip', '')).strip()
                    )
                    phone = str(row.get('phone', '')).strip()
                    existing_keys.add(('city_zip', city_zip_key))
                    if phone and phone != 'nan':
                        existing_keys.add(('phone', phone))
                
                # Check each new pharmacy against existing ones
                is_duplicate = []
                for _, row in new_pharmacies.iterrows():
                    # Check city and zip match
                    city_zip_key = (
                        str(row.get('city', '')).lower().strip(),
                        str(row.get('zip', '')).strip()
                    )
                    # Check phone match if available
                    phone = str(row.get('phone', '')).strip()
                    
                    # Consider it a duplicate if city+zip matches, or phone matches (if phone exists)
                    dup = ('city_zip', city_zip_key) in existing_keys
                    if not dup and phone and phone != 'nan':
                        dup = ('phone', phone) in existing_keys
                    
                    is_duplicate.append(dup)
                
                is_duplicate = pd.Series(is_duplicate, index=new_pharmacies.index)
                logger.info(f"Found {is_duplicate.sum()} duplicates out of {len(new_pharmacies)} new pharmacies")
                
                # Only add new pharmacies if they're not duplicates
                unique_new = new_pharmacies[~is_duplicate].head(count)
                
                # If we couldn't find enough unique pharmacies, don't add any
                if len(unique_new) < count:
                    logger.info(f"Could not find enough unique pharmacies. Found {len(unique_new)}, needed {count}. Returning existing pharmacies.")
                    return result
                
                if not unique_new.empty:
                    result = pd.concat([result, unique_new], ignore_index=True)
                    logger.info(f"Added {len(unique_new)} new unique pharmacies to {state}")
                else:
                    logger.info(f"No new unique pharmacies found for {state}. Returning existing pharmacies.")
                
                return result
            else:
                # If no existing pharmacies, just add the new ones up to count
                new_pharmacies = new_pharmacies.head(count)
                result = pd.concat([result, new_pharmacies], ignore_index=True)
                logger.info(f"Added {len(new_pharmacies)} new pharmacies to {state}")
    
    return result

def get_cities_for_state(state: str) -> List[str]:
    """
    Get a list of major cities for a given state.
    
    Args:
        state: Two-letter state code
        
    Returns:
        List of city names in the state
    """
    # A simple mapping of states to their major cities
    # In a production environment, this could be replaced with a more comprehensive list
    # or an API call to a geocoding service
    state_cities = {
        'AL': ['Birmingham', 'Montgomery', 'Mobile', 'Huntsville', 'Tuscaloosa'],
        'AK': ['Anchorage', 'Fairbanks', 'Juneau', 'Sitka', 'Wasilla'],
        'AZ': ['Phoenix', 'Tucson', 'Mesa', 'Chandler', 'Scottsdale'],
        'AR': ['Little Rock', 'Fort Smith', 'Fayetteville', 'Springdale', 'Jonesboro'],
        'CA': ['Los Angeles', 'San Diego', 'San Jose', 'San Francisco', 'Fresno'],
        'CO': ['Denver', 'Colorado Springs', 'Aurora', 'Fort Collins', 'Lakewood'],
        'CT': ['Bridgeport', 'New Haven', 'Stamford', 'Hartford', 'Waterbury'],
        'DE': ['Wilmington', 'Dover', 'Newark', 'Middletown', 'Smyrna'],
        'FL': ['Jacksonville', 'Miami', 'Tampa', 'Orlando', 'St. Petersburg'],
        'GA': ['Atlanta', 'Augusta', 'Columbus', 'Savannah', 'Athens'],
        'HI': ['Honolulu', 'East Honolulu', 'Pearl City', 'Hilo', 'Kailua'],
        'ID': ['Boise', 'Meridian', 'Nampa', 'Idaho Falls', 'Pocatello'],
        'IL': ['Chicago', 'Aurora', 'Naperville', 'Joliet', 'Rockford'],
        'IN': ['Indianapolis', 'Fort Wayne', 'Evansville', 'South Bend', 'Carmel'],
        'IA': ['Des Moines', 'Cedar Rapids', 'Davenport', 'Sioux City', 'Iowa City'],
        'KS': ['Wichita', 'Overland Park', 'Kansas City', 'Olathe', 'Topeka'],
        'KY': ['Louisville', 'Lexington', 'Bowling Green', 'Owensboro', 'Covington'],
        'LA': ['New Orleans', 'Baton Rouge', 'Shreveport', 'Lafayette', 'Lake Charles'],
        'ME': ['Portland', 'Lewiston', 'Bangor', 'South Portland', 'Auburn'],
        'MD': ['Baltimore', 'Frederick', 'Rockville', 'Gaithersburg', 'Bowie'],
        'MA': ['Boston', 'Worcester', 'Springfield', 'Lowell', 'Cambridge'],
        'MI': ['Detroit', 'Grand Rapids', 'Warren', 'Sterling Heights', 'Ann Arbor'],
        'MN': ['Minneapolis', 'St. Paul', 'Rochester', 'Duluth', 'Bloomington'],
        'MS': ['Jackson', 'Gulfport', 'Southaven', 'Hattiesburg', 'Biloxi'],
        'MO': ['Kansas City', 'St. Louis', 'Springfield', 'Columbia', 'Independence'],
        'MT': ['Billings', 'Missoula', 'Great Falls', 'Bozeman', 'Butte'],
        'NE': ['Omaha', 'Lincoln', 'Bellevue', 'Grand Island', 'Kearney'],
        'NV': ['Las Vegas', 'Henderson', 'Reno', 'North Las Vegas', 'Sparks'],
        'NH': ['Manchester', 'Nashua', 'Concord', 'Dover', 'Rochester'],
        'NJ': ['Newark', 'Jersey City', 'Paterson', 'Elizabeth', 'Clifton'],
        'NM': ['Albuquerque', 'Las Cruces', 'Rio Rancho', 'Santa Fe', 'Roswell'],
        'NY': ['New York', 'Buffalo', 'Rochester', 'Syracuse', 'Albany'],
        'NC': ['Charlotte', 'Raleigh', 'Greensboro', 'Durham', 'Winston-Salem'],
        'ND': ['Fargo', 'Bismarck', 'Grand Forks', 'Minot', 'West Fargo'],
        'OH': ['Columbus', 'Cleveland', 'Cincinnati', 'Toledo', 'Akron'],
        'OK': ['Oklahoma City', 'Tulsa', 'Norman', 'Broken Arrow', 'Lawton'],
        'OR': ['Portland', 'Salem', 'Eugene', 'Gresham', 'Hillsboro'],
        'PA': ['Philadelphia', 'Pittsburgh', 'Allentown', 'Erie', 'Reading'],
        'RI': ['Providence', 'Warwick', 'Cranston', 'Pawtucket', 'East Providence'],
        'SC': ['Columbia', 'Charleston', 'North Charleston', 'Mount Pleasant', 'Rock Hill'],
        'SD': ['Sioux Falls', 'Rapid City', 'Aberdeen', 'Brookings', 'Watertown'],
        'TN': ['Nashville', 'Memphis', 'Knoxville', 'Chattanooga', 'Clarksville'],
        'TX': ['Houston', 'San Antonio', 'Dallas', 'Austin', 'Fort Worth'],
        'UT': ['Salt Lake City', 'West Valley City', 'Provo', 'West Jordan', 'Orem'],
        'VT': ['Burlington', 'South Burlington', 'Rutland', 'Barre', 'Montpelier'],
        'VA': ['Virginia Beach', 'Norfolk', 'Chesapeake', 'Richmond', 'Newport News'],
        'WA': ['Seattle', 'Spokane', 'Tacoma', 'Vancouver', 'Bellevue'],
        'WV': ['Charleston', 'Huntington', 'Morgantown', 'Parkersburg', 'Wheeling'],
        'WI': ['Milwaukee', 'Madison', 'Green Bay', 'Kenosha', 'Racine'],
        'WY': ['Cheyenne', 'Casper', 'Laramie', 'Gillette', 'Rock Springs'],
        'DC': ['Washington']
    }
    
    # Return the cities for the given state, or an empty list if state not found
    return state_cities.get(state.upper(), [state])

def get_major_cities_for_state(state: str) -> List[str]:
    """
    Get a list of major cities for a given state.
    
    Args:
        state: Two-letter state code
        
    Returns:
        List of city names
    """
    # This is a simplified version - in production, you might want to use a more comprehensive list
    # or an API call to a geocoding service
    major_cities = {
        'CA': ['Los Angeles', 'San Francisco', 'San Diego', 'San Jose', 'Sacramento'],
        'TX': ['Houston', 'Dallas', 'Austin', 'San Antonio', 'Fort Worth'],
        'NY': ['New York', 'Buffalo', 'Rochester', 'Syracuse', 'Albany'],
        'FL': ['Miami', 'Orlando', 'Tampa', 'Jacksonville', 'Tallahassee'],
        'IL': ['Chicago', 'Springfield', 'Peoria', 'Rockford', 'Naperville']
    }
    
    return major_cities.get(state.upper(), [])

def merge_new_pharmacies(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """
    Merge new pharmacies with existing ones, removing duplicates.
    
    Args:
        existing: DataFrame of existing pharmacies
        new: DataFrame of new pharmacies to add
        
    Returns:
        DataFrame with merged pharmacies, with duplicates removed (keeping highest confidence)
        and maintaining original index from existing DataFrame
   """
    if existing.empty:
        return new.copy()
    
    if new.empty:
        return existing.copy()
    
    # Ensure we have required columns
    required_columns = ['name', 'address', 'city', 'state', 'zip', 'phone', 'is_chain', 'confidence']
    
    # Make a clean copy of existing with reset index
    existing_clean = existing[required_columns].copy().reset_index(drop=True)
    
    # Make a clean copy of new
    new = new[required_columns].copy()
    
    # Clean phone numbers for comparison
    def clean_phone(phone):
        if pd.isna(phone):
            return ''
        # Remove all non-digit characters
        return ''.join(filter(str.isdigit, str(phone)))
    
    # Clean phone numbers in both DataFrames
    existing_clean['phone'] = existing_clean['phone'].apply(clean_phone)
    new['phone'] = new['phone'].apply(clean_phone)
    
    # Create a matching key for fuzzy matching
    def create_match_key(row):
        name = str(row['name']).lower().strip()
        city = str(row['city']).lower().strip()
        zip_code = str(row['zip'])[:5]
        return f"{name[:10]}_{city[:5]}_{zip_code}"
    
    # Add matching key
    existing_clean['match_key'] = existing_clean.apply(create_match_key, axis=1)
    new['match_key'] = new.apply(create_match_key, axis=1)
    
    # Find duplicates between existing and new
    duplicates_mask = new['match_key'].isin(existing_clean['match_key'])
    
    # For non-duplicates, add to existing
    new_unique = new[~duplicates_mask].copy()
    
    # For duplicates, check if we should replace based on confidence
    for _, dup_row in new[duplicates_mask].iterrows():
        match_key = dup_row['match_key']
        existing_idx = existing_clean[existing_clean['match_key'] == match_key].index[0]
        
        # Replace if new row has higher confidence
        if dup_row['confidence'] > existing_clean.loc[existing_idx, 'confidence']:
            existing_clean.loc[existing_idx] = dup_row[required_columns]
    
    # Combine existing and new unique pharmacies
    result = pd.concat([existing_clean, new_unique], ignore_index=True)
    
    # Clean up temporary columns
    result = result[required_columns]
    
    # Reset index to maintain original behavior
    result.reset_index(drop=True, inplace=True)
    
    return result

def process_pharmacies(pharmacies: pd.DataFrame, 
                     min_required: int = DEFAULT_MIN_REQUIRED) -> Dict[str, pd.DataFrame]:
    """
    Process pharmacies to ensure each state has enough unique pharmacies.
    
    Args:
        pharmacies: DataFrame of pharmacies to process
        min_required: Minimum number of pharmacies required per state
        
    Returns:
        Dictionary mapping state codes to DataFrames of processed pharmacies
    """
    if pharmacies.empty:
        return {}
    
    # Group by state and process each state separately
    grouped_pharmacies = {state: df for state, df in pharmacies.groupby('state')}
    
    # Identify which states need more pharmacies
    underfilled = identify_underfilled_states(grouped_pharmacies, min_required)
    
    # Process each underfilled state
    for state, additional_needed in underfilled.items():
        logger.info(f"State {state} needs {additional_needed} more pharmacies")
        
        # Get existing pharmacies for this state, or empty DataFrame if none
        existing = grouped_pharmacies.get(state, pd.DataFrame())
        
        # Try to get more pharmacies for this state
        new_pharmacies = self_heal_state(state, existing, additional_needed, min_required)
        
        if not new_pharmacies.empty:
            # Merge new pharmacies with existing ones
            if existing.empty:
                grouped_pharmacies[state] = new_pharmacies
            else:
                # Use merge_new_pharmacies to handle duplicates
                grouped_pharmacies[state] = merge_new_pharmacies(existing, new_pharmacies)
    
    # Convert all DataFrames to the expected format
    result = {}
    for state, df in grouped_pharmacies.items():
        # Ensure consistent column order and types
        result[state] = df[['name', 'address', 'city', 'state', 'zip', 'phone', 'is_chain', 'confidence']].copy()
        # Reset index to ensure clean output
        result[state].reset_index(drop=True, inplace=True)
    
    return result
