"""Main entry point for running the pharmacy scraper pipeline (V2).

This script uses the resilient PipelineOrchestrator to manage and execute
the data collection, processing, and classification tasks. It supports
automatic resumption from the last failed or incomplete stage.

Example:
    # Run the pipeline using a specific configuration
    python -m src.pharmacy_scraper.run_pipeline_v2 --config config/pipeline_config.json

    # Reset the pipeline state and start a fresh run
    python -m src.pharmacy_scraper.run_pipeline_v2 --config config/pipeline_config.json --reset
"""

import argparse
import logging
from pathlib import Path

from src.pharmacy_scraper.orchestrator.pipeline_orchestrator import PipelineOrchestrator

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s (%(filename)s:%(lineno)d)",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the pipeline runner."""
    parser = argparse.ArgumentParser(
        description="Run the pharmacy scraper pipeline with resume capabilities."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the JSON configuration file for the pipeline."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset the pipeline state and start a fresh run. Deletes the existing state database."
    )
    return parser.parse_args()


def main():
    """Main function to initialize and run the pipeline orchestrator."""
    args = _parse_args()
    config_path = Path(args.config)

    if not config_path.is_file():
        logger.error(f"Configuration file not found at: {config_path}")
        return

    try:
        orchestrator = PipelineOrchestrator(config_path=str(config_path))

        if args.reset:
            logger.info("Resetting pipeline state as requested.")
            orchestrator.state_manager.reset_state()

        logger.info("Starting pipeline orchestrator...")
        final_output_path = orchestrator.run()

        if final_output_path:
            logger.info(f"✅ Pipeline completed successfully. Final output at: {final_output_path}")
        else:
            logger.warning("Pipeline did not complete successfully. See logs for details.")

    except Exception as e:
        logger.critical(f"❌ A critical error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()
