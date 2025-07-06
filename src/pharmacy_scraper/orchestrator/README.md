# Orchestrator Module

The Orchestrator module is responsible for managing the entire pipeline of the Pharmacy Scraper system, coordinating the flow between different stages including data collection, deduplication, classification, and verification.

## Overview

The Orchestrator provides a robust pipeline execution framework with the following key capabilities:

- **Pipeline State Management**: Tracks the progress of pipeline execution and allows for resumable operations
- **Intelligent Caching**: Minimizes redundant API calls by caching query results
- **API Budget Management**: Monitors and controls API usage to stay within defined limits
- **Error Handling**: Graceful error recovery and detailed logging
- **Stage Execution**: Manages the sequence and dependencies between pipeline stages

## Key Components

### PipelineOrchestrator

The main class that coordinates the entire pipeline execution.

```python
from pharmacy_scraper.orchestrator import PipelineOrchestrator
from pharmacy_scraper.config import PipelineConfig

config = PipelineConfig()
orchestrator = PipelineOrchestrator(config)
results = orchestrator.run(queries=["pharmacy"], locations=["San Francisco, CA"])
```

### State Manager

Handles the persistence and management of pipeline state, enabling resumable operations.

```python
from pharmacy_scraper.orchestrator.state_manager import StateManager

state_manager = StateManager(output_dir="./output")
state = state_manager.load_state()
# Check if a stage has been completed
if state.is_stage_completed("collection"):
    print("Collection stage is already complete")
```

## Cache Functionality

The Orchestrator implements a robust caching system to optimize API usage and improve performance:

### Features

- **Query-based Caching**: Results are cached based on the query and location
- **Automatic Directory Creation**: Cache directories are created if they don't exist
- **Invalid Cache Handling**: Corrupted cache files are detected and regenerated
- **Cache Keys**: Standardized cache key format (lowercase, underscores)

### Example

```python
# Cache is automatically managed by the orchestrator
orchestrator = PipelineOrchestrator(config)

# First run - will query APIs and cache results
results1 = orchestrator.run(queries=["pharmacy"], locations=["San Francisco, CA"])

# Second run - will use cached results without making API calls
results2 = orchestrator.run(queries=["pharmacy"], locations=["San Francisco, CA"])
```

## Test Coverage

The orchestrator module has comprehensive test coverage (88%) with specific tests for:

1. **Cache Hit Testing**: Verifies proper usage of cached results
2. **Cache Miss Testing**: Ensures API calls are made when cache is missing
3. **Invalid Cache Handling**: Tests recovery from corrupted cache files
4. **Cache Directory Creation**: Confirms automatic creation of cache directories
5. **Pipeline Resumability**: Tests the ability to resume pipeline from intermediate states

## Dependencies

- `pharmacy_scraper.api.apify_collector`: For data collection
- `pharmacy_scraper.classification.classifier`: For pharmacy classification
- `pharmacy_scraper.verification.google_places`: For data verification
- `pharmacy_scraper.utils.cache`: For caching operations
- `pharmacy_scraper.utils.api_usage_tracker`: For API budget management
