"""Package for configuration-related modules."""

import os
from typing import Dict, Optional

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not installed; skip loading .env files
    pass

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

__all__ = ['get_config']
