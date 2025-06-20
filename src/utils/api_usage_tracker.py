"""
API Usage Tracker - Track and limit API usage to manage credits.
"""
import os
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CreditLimitExceededError(Exception):
    """Raised when the credit limit would be exceeded by an API call."""
    pass

class APICreditTracker:
    """Track and manage API credit usage."""
    
    def __init__(self, 
                 budget: float = 100.0,  # Default budget in credits
                 daily_limit: Optional[float] = None,  # Daily limit in credits
                 cache_dir: str = '.api_cache'):
        """Initialize the credit tracker.
        
        Args:
            budget: Total credit budget for the project
            daily_limit: Maximum credits to use per day
            cache_dir: Directory to store usage data
        """
        self.budget = budget
        self.daily_limit = daily_limit or budget / 4  # Default to 1/4 of total budget per day
        self.cache_dir = Path(cache_dir)
        self.usage_file = self.cache_dir / 'api_usage.json'
        self.usage_data = self._load_usage_data()
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_usage_data(self) -> Dict:
        """Load usage data from file or initialize if it doesn't exist."""
        if self.usage_file.exists():
            try:
                with open(self.usage_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load usage data: {e}")
        
        # Initialize with default structure
        return {
            'total_used': 0.0,
            'daily_usage': {},
            'last_updated': datetime.utcnow().isoformat()
        }
    
    def _save_usage_data(self):
        """Save current usage data to file."""
        try:
            with open(self.usage_file, 'w') as f:
                json.dump(self.usage_data, f, indent=2)
        except IOError as e:
            logger.error(f"Failed to save usage data: {e}")
    
    def _get_today_key(self) -> str:
        """Get today's date key in YYYY-MM-DD format."""
        return datetime.utcnow().strftime('%Y-%m-%d')
    
    def _get_daily_used(self) -> float:
        """Get credits used today."""
        today = self._get_today_key()
        return self.usage_data['daily_usage'].get(today, 0.0)
    
    def check_credit_available(self, estimated_cost: float = 1.0) -> bool:
        """Check if there are enough credits available.
        
        Args:
            estimated_cost: Estimated cost of the upcoming API call
            
        Returns:
            bool: True if credits are available, False otherwise
        """
        total_used = self.usage_data['total_used']
        daily_used = self._get_daily_used()
        
        # Check budget limits
        if (total_used + estimated_cost) > self.budget:
            logger.warning(f"Budget exceeded: {total_used}/{self.budget} credits used")
            return False
            
        if (daily_used + estimated_cost) > self.daily_limit:
            logger.warning(f"Daily limit reached: {daily_used}/{self.daily_limit} credits used today")
            return False
            
        return True
    
    def record_usage(self, credits_used: float, operation: str = "api_call") -> None:
        """Record API usage.
        
        Args:
            credits_used: Number of credits used
            operation: Description of the operation
        """
        today = self._get_today_key()
        
        # Update totals
        self.usage_data['total_used'] += credits_used
        self.usage_data['daily_usage'][today] = self.usage_data['daily_usage'].get(today, 0) + credits_used
        self.usage_data['last_updated'] = datetime.utcnow().isoformat()
        
        # Log the usage
        logger.info(f"API usage recorded: {credits_used:.2f} credits for {operation}")
        logger.info(f"Total used: {self.usage_data['total_used']:.2f}/{self.budget:.2f}")
        logger.info(f"Today's usage: {self.usage_data['daily_usage'].get(today, 0):.2f}/{self.daily_limit:.2f}")
        
        # Save to disk
        self._save_usage_data()
    
    def get_usage_summary(self) -> Dict[str, Union[float, Dict]]:
        """Get a summary of API usage."""
        return {
            'total_used': self.usage_data['total_used'],
            'total_budget': self.budget,
            'remaining': self.budget - self.usage_data['total_used'],
            'daily_limit': self.daily_limit,
            'today_used': self._get_daily_used(),
            'last_updated': self.usage_data['last_updated']
        }

# Global instance for easy access
try:
    # Try to get budget from environment variable or use default
    BUDGET = float(os.getenv('API_BUDGET', '100.0'))
    credit_tracker = APICreditTracker(budget=BUDGET)
except ValueError:
    logger.warning("Invalid API_BUDGET, using default 100.0")
    credit_tracker = APICreditTracker()
