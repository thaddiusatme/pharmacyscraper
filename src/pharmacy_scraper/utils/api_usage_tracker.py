"""
API Usage Tracker - Track and limit API usage to manage credits.
"""
import os
import time
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
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
            'last_updated': datetime.now(timezone.utc).isoformat()
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
        return datetime.now(timezone.utc).strftime('%Y-%m-%d')
    
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
    
    def track_usage(self, service: str, cost: float = 1.0):
        """Context manager to track API usage with automatic credit deduction.
        
        Args:
            service: Name of the service being used (e.g., 'apify', 'google_places')
            cost: Cost in credits for this API call
            
        Yields:
            None
            
        Raises:
            CreditLimitExceededError: If the cost would exceed the credit limit
        """
        if not self.check_credit_available(cost):
            raise CreditLimitExceededError(
                f"Insufficient credits for {service} operation (cost: {cost}). "
                f"Used {self.usage_data['total_used']}/{self.budget} total, "
                f"{self._get_daily_used()}/{self.daily_limit} today."
            )
            
        class CreditTrackerContext:
            def __init__(self, parent, service, cost):
                self.parent = parent
                self.service = service
                self.cost = cost
                self.start_time = time.time()
                
            def __enter__(self):
                self.parent.record_usage(self.cost, f"{self.service}_api_call")
                logger.debug(f"Started {self.service} API call (cost: {self.cost})")
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                duration = time.time() - self.start_time
                if exc_type is not None:
                    logger.error(
                        f"{self.service} API call failed after {duration:.2f}s: {exc_val}",
                        exc_info=(exc_type, exc_val, exc_tb)
                    )
                else:
                    logger.debug(f"Completed {self.service} API call in {duration:.2f}s")
                return False  # Don't suppress exceptions
                
        return CreditTrackerContext(self, service, cost)
        
    def set_cost_limit(self, service: str, cost: float) -> None:
        """Set the cost limit for a specific service.
        
        Args:
            service: Name of the service (e.g., 'apify', 'google_places')
            cost: Cost per API call for this service
        """
        if not hasattr(self, '_service_limits'):
            self._service_limits = {}
        self._service_limits[service] = cost
        logger.debug(f"Set cost limit for {service}: ${cost:.4f} per call")
        
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
        self.usage_data['last_updated'] = datetime.now(timezone.utc).isoformat()
        
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
    
    def get_total_usage(self) -> float:
        """Get the total number of credits used.
        
        Returns:
            float: Total credits used
        """
        return self.usage_data.get('total_used', 0.0)

    def reset(self):
        """Reset usage data to its initial state, for testing purposes."""
        self.usage_data = {
            'total_used': 0.0,
            'daily_usage': {},
            'last_updated': datetime.now(timezone.utc).isoformat()
        }
        # Overwrite the usage file with the reset state
        self._save_usage_data()
        logger.info("API credit tracker has been reset.")

# Global instance for easy access
try:
    # Try to get budget from environment variable or use default
    BUDGET = float(os.getenv('API_BUDGET', '100.0'))
    credit_tracker = APICreditTracker(budget=BUDGET)
except ValueError:
    logger.warning("Invalid API_BUDGET, using default 100.0")
    credit_tracker = APICreditTracker()
