Stabilize core and tests [critical]

 Fix pandas issues: ensure real pandas is imported and not shadowed; restore pd.DataFrame isinstance checks and pd.concat usage
 Orchestrator: add/restore StateManager.get_state() to satisfy tests
 PerplexityClient: align default model (single source of truth) and retry exceptions so tests actually retry
 Dedup/self-heal: decide contract (raise vs. graceful handling) and align tests accordingly
 Re-run full test suite; aim for 0 errors/failures (keep coverage ≥ 67% now; stretch 80%)
 Lock dependencies: add/refresh requirements with pinned versions known to pass
Configuration-driven groundwork [critical]

 Define config schema (YAML/JSON) for: search_terms, regions, batch size, budgets
 Add schema validation with clear errors (e.g., pydantic/voluptuous)
 Provide example configs and minimal docs (docs/config/)
Plugin architecture prep

 Define classifier plugin interface (e.g., BaseClassifierPlugin with hooks: chain detection, compounding, post-processing)
 Implement plugin registry/loader (entry points or module discovery)
 Add example plugin and tests
Caching and idempotency

 Standardize deterministic cache keys and locations
 Define invalidation strategy and “force_reclassification” behavior
 Add cache hit/miss metrics
API cost caps and safety

 Implement per-search-term budget caps with fail-safe cutoff
 Wire into API usage tracker and surface in logs/metrics
Data pipeline contracts

 Document required columns (e.g., state) and expected behaviors
 Make CSV merge and dedupe steps deterministic; clarify missing column handling
 Update/align fixtures and integration tests
Observability and diagnostics

 Structured logging (run_id, stage, source, cost, cache_source)
 Minimal metrics: calls, retries, cache hits, cost per term/region
 Debug toggles and redaction for sensitive fields
Security and secrets

 Verify secrets never logged; redact by default
 Confirm .gitignore excludes cache/, artifacts, and local env
 Enable/validate secrets scanning (commit hooks or CI)
CI/CD and workflows

 GitHub Actions: run pytest, collect coverage, enforce conventional commits
 Set minimal coverage gate at current 67% (target 80% later)
 Branch protection: required checks before merge
Rules and workflow sync

 Sync key .windsurf rule updates into .cascade/rules/pharmacy_scraper.yaml for consistency
 Add integration test reminders for workflow-affecting changes
Documentation

 Update Epic with agnostic plan, milestones, and risks
 ADR: configuration-driven + plugin architecture decisions
 README quickstart for config-driven runs
Rollout plan

 Feature flag: enable agnostic workflow side-by-side with current
 Migration path for existing runs and caches
 Success criteria: green tests, budget enforcement works, plugins load, config-only runs succeed
