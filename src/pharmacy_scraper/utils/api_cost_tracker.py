"""API cost tracking and calculation utilities."""
import logging
from typing import Any, Dict, Optional, Union, List, TypeVar, Type, cast

from .api_usage_tracker import APICreditTracker

logger = logging.getLogger(__name__)
T = TypeVar('T')

class APICostTracker:
    """Track and calculate API costs based on actual API responses.
    
    This class provides functionality to calculate and record API usage costs
    for various API providers and services. It supports both built-in providers
    (Google Places, Apify, OpenAI) and custom cost structures.
    
    Usage:
        # Record usage with default cost structure
        cost = APICostTracker.record_usage_from_response(
            provider='google_places',
            service='details',
            response=api_response,
            credit_tracker=tracker
        )
        
        # Record usage with custom cost structure
        APICostTracker.COST_STRUCTURES['custom_provider'] = {
            'custom_service': {'base': 0.05, 'per_item': 0.0003}
        }
    """
    
    # Cost structures for different API providers
    COST_STRUCTURES = {
        'google_places': {
            'details': 0.02,  # Per details request
            'nearby_search': 0.0,  # Base cost per search
            'nearby_search_per_result': 0.032 / 1000,  # Per result in search
            'text_search': 0.01,  # Per text search
        },
        'apify': {
            'google_maps_scraper': {
                'base': 0.10,  # Base cost per run
                'per_result': 0.001,  # Per result
            }
        },
        'openai': {
            'gpt-4': {
                'input': 0.03,  # Per 1K tokens
                'output': 0.06,  # Per 1K tokens
            },
            'gpt-3.5-turbo': {
                'input': 0.0015,  # Per 1K tokens
                'output': 0.002,  # Per 1K tokens
            }
        }
    }

    @classmethod
    def record_usage_from_response(
        cls,
        provider: str,
        service: str,
        response: Any,
        operation: str = "api_call",
        credit_tracker: Optional[APICreditTracker] = None,
        **kwargs
    ) -> float:
        """Record API usage based on the actual response.
        
        Args:
            provider: API provider (e.g., 'google_places', 'apify', 'openai')
            service: Service name (e.g., 'details', 'gpt-4')
            response: API response object
            operation: Operation name for logging (default: 'api_call')
            credit_tracker: Optional APICreditTracker instance to record usage
            **kwargs: Additional arguments for cost calculation
            
        Returns:
            float: Calculated cost in credits
            
        Example:
            >>> cost = APICostTracker.record_usage_from_response(
            ...     provider='google_places',
            ...     service='details',
            ...     response={'result': {}},
            ...     credit_tracker=my_tracker
            ... )
        """
        try:
            cost = cls._calculate_cost(provider, service, response, **kwargs)
            
            if credit_tracker is not None:
                credit_tracker.record_usage(cost, f"{provider}.{service}.{operation}")
            return cost
            
        except Exception as e:
            logger.error(f"Error calculating API cost: {str(e)}", exc_info=True)
            # Fall back to default cost on error
            fallback_cost = 0.01
            if credit_tracker is not None:
                credit_tracker.record_usage(
                    fallback_cost, 
                    f"{provider}.{service}.{operation}_fallback"
                )
            return fallback_cost

    @classmethod
    def _calculate_cost(
        cls,
        provider: str,
        service: str,
        response: Any,
        **kwargs
    ) -> float:
        """Calculate cost based on provider, service, and response."""
        # Try built-in providers first
        if provider == 'google_places':
            return cls._calculate_google_places_cost(service, response, **kwargs)
        elif provider == 'apify':
            return cls._calculate_apify_cost(service, response, **kwargs)
        elif provider == 'openai':
            return cls._calculate_openai_cost(service, response, **kwargs)
            
        # Check for custom cost structure
        if provider in cls.COST_STRUCTURES and service in cls.COST_STRUCTURES[provider]:
            service_costs = cls.COST_STRUCTURES[provider][service]
            if isinstance(service_costs, dict):
                return cls._calculate_structured_cost(service_costs, response)
            return float(service_costs)  # Simple fixed cost
            
        # Default cost for unknown providers
        logger.warning(f"Using default cost for unknown provider: {provider}.{service}")
        return 0.01

    @classmethod
    def _calculate_structured_cost(
        cls,
        cost_structure: Dict[str, float],
        response: Any
    ) -> float:
        """Calculate cost using a structured cost definition."""
        base_cost = cost_structure.get('base', 0.0)
        if 'per_item' in cost_structure:
            item_count = cls._get_item_count(response)
            return base_cost + (item_count * cost_structure['per_item'])
        return base_cost

    @staticmethod
    def _get_item_count(response: Any) -> int:
        """Extract item count from various response formats."""
        if hasattr(response, 'get') and callable(response.get):
            if 'items' in response:
                return len(response['items'])
            if isinstance(response, (dict, list, tuple)):
                return len(response)
        elif hasattr(response, 'items'):
            return len(response.items)
        elif isinstance(response, (list, tuple)):
            return len(response)
        return 0

    @classmethod
    def _calculate_google_places_cost(
        cls, 
        service: str, 
        response: Any, 
        **kwargs
    ) -> float:
        """Calculate cost for Google Places API calls.
        
        Args:
            service: The Google Places service being used (e.g., 'details', 'nearby_search')
            response: The API response object
            **kwargs: Additional arguments for cost calculation
            
        Returns:
            float: The calculated cost in credits
        """
        costs = cls.COST_STRUCTURES.get('google_places', {})
        
        if service == 'details':
            return costs.get('details', 0.02)
            
        if service == 'nearby_search':
            # For nearby_search, look for the 'results' key in the response
            if hasattr(response, 'get') and callable(response.get):
                num_results = len(response.get('results', []))
            elif hasattr(response, 'results'):
                num_results = len(response.results)
            else:
                num_results = 0
                
            # Use the per-result cost from the cost structure
            per_result = costs.get('nearby_search_per_result', 0.000032)  # 0.032 / 1000
            return num_results * per_result
            
        if service == 'text_search':
            return costs.get('text_search', 0.01)
            
        return 0.01  # Default cost for unknown services

    @classmethod
    def _calculate_apify_cost(
        cls, 
        service: str, 
        response: Any, 
        **kwargs
    ) -> float:
        """Calculate cost for Apify actor runs."""
        service_costs = cls.COST_STRUCTURES.get('apify', {}).get(service, {})
        if not service_costs:
            return 0.01
            
        base_cost = service_costs.get('base', 0.10)
        per_result = service_costs.get('per_result', 0.001)
        item_count = cls._get_item_count(response)
        
        return base_cost + (item_count * per_result)

    @classmethod
    def _calculate_openai_cost(
        cls, 
        model: str, 
        response: Any, 
        **kwargs
    ) -> float:
        """Calculate cost for OpenAI API calls."""
        model_costs = cls.COST_STRUCTURES.get('openai', {}).get(model, {})
        if not model_costs:
            return 0.01
            
        # Get token counts from response
        usage = getattr(response, 'usage', {})
        if hasattr(usage, 'get') and callable(usage.get):
            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)
        else:
            prompt_tokens = getattr(usage, 'prompt_tokens', 0)
            completion_tokens = getattr(usage, 'completion_tokens', 0)
            
        # Calculate cost per token
        input_cost = (prompt_tokens / 1000) * model_costs.get('input', 0.0)
        output_cost = (completion_tokens / 1000) * model_costs.get('output', 0.0)
        
        return input_cost + output_cost
