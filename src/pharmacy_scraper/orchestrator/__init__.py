"""
Orchestrator module for coordinating the pharmacy scraper pipeline.

This module provides the main PipelineOrchestrator class that manages the entire
pharmacy data collection, processing, and verification workflow.

Example:
    >>> from pharmacy_scraper.orchestrator import PipelineOrchestrator
    >>> orchestrator = PipelineOrchestrator(config_path="config.json")
    >>> orchestrator.run()
"""

from .pipeline_orchestrator import PipelineOrchestrator

__all__ = ["PipelineOrchestrator"]
