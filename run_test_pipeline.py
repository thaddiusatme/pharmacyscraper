#!/usr/bin/env python
"""
Test runner for the pharmacy scraper pipeline with mocked external services.
This script patches the external API calls to safely run the pipeline without real API keys.
"""

import logging
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.pharmacy_scraper.orchestrator.pipeline_orchestrator import PipelineOrchestrator
from src.pharmacy_scraper.utils.api_usage_tracker import credit_tracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)8s] %(message)s (%(filename)s:%(lineno)d)",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def main():
    """Run the pipeline with mocked services for testing."""
    # Reset credit tracker to ensure a clean state
    credit_tracker.reset()
    
    # Config path
    config_path = Path("config/test/small_sample_test.json")
    
    # Sample data for mocking
    raw_data = [
        {"name": "Pharmacy A", "address": "123 Main St", "city": "Phoenix", "state": "AZ"},
        {"name": "Pharmacy B", "address": "456 Oak St", "city": "Phoenix", "state": "AZ"},
        {"name": "Pharmacy C", "address": "789 Pine St", "city": "Phoenix", "state": "AZ"},
    ]
    
    # Create and run the orchestrator with mocked methods
    with patch('src.pharmacy_scraper.orchestrator.pipeline_orchestrator.ApifyCollector') as mock_collector_class:
        # Configure collector mock
        mock_collector = mock_collector_class.return_value
        mock_collector.run_trial.return_value = raw_data
        
        # Initialize orchestrator
        orchestrator = PipelineOrchestrator(config_path=str(config_path))
        
        # Reset the state manager to ensure we start fresh
        logger.info("Resetting pipeline state to start fresh")
        orchestrator.state_manager.reset_state()
        
        # Replace verification method with mock to avoid real API calls
        original_verify = orchestrator._verify_pharmacies
        def mock_verify_pharmacies(pharmacies):
            logger.info(f"Using mock verify_pharmacies method on {len(pharmacies)} pharmacies")
            # Add verification data to each pharmacy
            return [{**p, "verification": {"verified": True, "confidence": 0.95}} for p in pharmacies]
        orchestrator._verify_pharmacies = mock_verify_pharmacies
        
        # Replace classification method with mock
        original_classify = orchestrator._classify_pharmacies
        def mock_classify_pharmacies(pharmacies):
            logger.info(f"Using mock classify_pharmacies method on {len(pharmacies)} pharmacies")
            # Add classification data to each pharmacy
            return [{**p, "classification": {"is_pharmacy": True, "confidence": 0.9}} for p in pharmacies]
        orchestrator._classify_pharmacies = mock_classify_pharmacies
        
        try:
            logger.info("Starting pipeline run with mocked services...")
            result = orchestrator.run()
            
            if result:
                logger.info(f"✅ Pipeline completed successfully. Final output at: {result}")
                # Display content of the output file
                import json
                with open(result, 'r') as f:
                    data = json.load(f)
                    logger.info(f"Output contains {len(data)} pharmacies")
                    for item in data[:3]:  # Show up to 3 items
                        logger.info(f"  - {item['name']}: {item['address']}")
            else:
                logger.warning("❌ Pipeline did not complete successfully.")
                
        except Exception as e:
            logger.critical(f"❌ Error running pipeline: {e}", exc_info=True)
            
        finally:
            # Restore original methods
            orchestrator._verify_pharmacies = original_verify
            orchestrator._classify_pharmacies = original_classify

if __name__ == "__main__":
    main()
