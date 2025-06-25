"""
Configuration management for the Pharmacy Scrape project.

This module handles loading configuration from environment variables
and a .env file.
"""

import os
from dotenv import load_dotenv
from typing import Dict, Optional

# Load environment variables from .env file
load_dotenv()

_config: Optional[Dict[str, str]] = None

def get_config() -> Dict[str, str]:
    """
    Loads configuration from environment variables and returns it as a dictionary.

    Caches the configuration after the first load to avoid reloading.

    Returns:
        A dictionary containing the configuration.
    """
    global _config
    if _config is None:
        _config = dict(os.environ)
    return _config
