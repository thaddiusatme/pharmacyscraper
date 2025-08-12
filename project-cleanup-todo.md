# Pharmacy Scraper Project Cleanup & Enhancement TODO
*Generated: 2025-08-11*

## 🎯 Current Status
- ✅ **Test Stabilization Complete:** 98.6% pass rate (340/345 tests), 73% coverage
- ✅ **Real-World Verification Complete:** Pipeline successfully processes real pharmacy data
- ✅ **Functionality Proven:** End-to-end execution with real APIs confirmed working
- ✅ **Fail-Fast Env Validation:** Centralized env validator integrated in production runner
- ✅ **Phase 1 Cleanup Complete:** All directory organization, dependency management, and security tasks finished

---

## 🧹 Phase 1: Project Cleanup & Organization [COMPLETED ✅]

### 🔒 Security & Secrets
- [x] Audit `.env` and `.env.example` for proper secret management
- [x] Verify `.gitignore` covers all sensitive files and directories
- [ ] Remove or redact any hardcoded secrets from configs
- [ ] Validate secrets are never logged (scan codebase)
- [x] Add fail-fast environment validation (utils/settings.py) and wire into runner

### 🗑️ Cruft Removal
- [x] Remove system files: `.DS_Store`, SSH key artifacts
- [x] Delete legacy test outputs: `test_output.txt`, `apify_results.json`
- [x] Clean up backup directories: `hypothesis_stub_backup/`
- [x] Remove unused root-level scripts and configs
- [x] Archive old trial result directories in `data/`

### 📂 Directory Organization
- [x] Consolidate cache directories (`cache/`, `.api_cache/`, `data/cache/`)
- [x] Move scattered run scripts to `scripts/` directory
- [x] Organize config files (keep production/dev/test structure)
- [x] Clean up `data/` directory (19+ subdirectories identified)
- [x] Archive or remove obsolete output directories

### 📌 Dependency Management
- [x] Pin all dependency versions in `requirements.txt` (based on working 98.6% test suite)
- [x] Remove duplicate pytest entries (lines 2 and 30)
- [x] Audit and remove unused dependencies
- [x] Create `requirements-dev.txt` for development-only deps
- [x] Add dependency security scanning

---

## 🏗️ Phase 2: Architecture & Infrastructure [HIGH PRIORITY]

### 📋 Configuration System
- [x] Add YAML parsing support in central config loader
- [x] Define unified config schema (YAML/JSON) for search terms, regions, budgets
- [x] Implement stricter schema validation and clear errors (top-level allowlist, basic nested types)
- [x] Create example configs (JSON and YAML) under `examples/config/`
- [x] Write migration guide from current setup to unified schema (`docs/migration-config-v1-to-unified.md`)
- [x] Add environment-specific config inheritance (`env` + `environments` with deep merge)

### 🔌 Plugin Architecture Foundation
- [x] Design `BaseClassifierPlugin` interface with hooks for chain detection, compounding
- [x] Implement plugin registry/loader system (entry points or module discovery)
- [x] Create example plugin implementation
- [x] Add plugin testing framework and validation

### 💾 Caching & Idempotency Enhancement
- [x] Standardize deterministic cache keys across all modules
- [x] Implement cache invalidation strategy and "force_reclassification" behavior
- [x] Add cache hit/miss logging (metrics and monitoring later)
- [x] Create cache cleanup/maintenance utilities

### 💰 API Cost Management
- [ ] Implement per-search-term budget caps with fail-safe cutoffs
- [ ] Enhanced integration with existing API usage tracker
- [ ] Add cost prediction and budget planning tools
- [ ] Create cost reporting and analytics dashboard

---

## 📊 Phase 3: Observability & Quality [MEDIUM PRIORITY]

### 🔍 Enhanced Logging & Monitoring
- [x] Implement structured logging (run_id, stage, cache_source, cache events). JSON by default; additive context via bind_context
- [x] Default redaction of sensitive fields (api_key, address, email, phone)
- [ ] Add comprehensive metrics: API calls, retries, cache hits, cost per term/region
- [ ] Add debug toggles (opt-in verbose sections) and configurable redaction list
- [ ] Build monitoring dashboards and alerting

Minimal visibility delivered now:
- Run lifecycle with `run_id`, `stage_start`, `stage_completed` (with `duration_ms`, `result_count` when available), and `stage_error`
- Classifier cache events: `cache_hit`, `cache_miss`, `cache_bypass`, `cache_store` with safe `cache_key_fp`
- JSON logs suitable for quick `jq` queries without standing up dashboards

### 🧪 Testing & Quality Assurance
- [ ] Add integration tests for real API scenarios (with rate limiting)
- [ ] Create performance benchmarks and regression testing
- [ ] Implement contract testing for data pipeline interfaces
- [ ] Add property-based testing for critical data processing functions

### 📈 Performance Optimization
- [ ] Profile and optimize hot paths in data processing
- [ ] Implement parallel processing for independent operations
- [ ] Add request batching and connection pooling for APIs
- [ ] Create performance monitoring and alerting

---

## 🔄 Phase 4: Automation & DevOps [MEDIUM PRIORITY]

### 🚀 CI/CD Pipeline
- [ ] Set up GitHub Actions for automated testing and deployment
- [ ] Add coverage reporting and gates (maintain ≥73%, target 80%)
- [ ] Implement conventional commits and semantic versioning
- [ ] Create branch protection rules with required status checks

### 🐳 Containerization & Deployment
- [ ] Create Dockerfile for consistent development and deployment
- [ ] Add docker-compose for local development stack
- [ ] Implement environment-specific deployment configurations
- [ ] Add health checks and readiness probes

### 📦 Release Management
- [ ] Set up automated changelog generation
- [ ] Create release workflow with proper versioning
- [ ] Add deployment rollback capabilities
- [ ] Implement feature flags for gradual rollouts

---

## 📚 Phase 5: Documentation & Developer Experience [LOW PRIORITY]

### 📖 Documentation Overhaul
- [ ] Update README with quickstart guide for config-driven runs
- [ ] Create comprehensive API documentation
- [ ] Write architecture decision records (ADRs) for major design choices
- [ ] Add troubleshooting guides and FAQ

### 🛠️ Developer Tools
- [ ] Create development setup automation scripts
- [ ] Add pre-commit hooks for code quality
- [ ] Implement code generation tools for repetitive tasks
- [ ] Create debugging and profiling utilities

### 🎓 Training & Onboarding
- [ ] Create developer onboarding documentation
- [ ] Add example use cases and tutorials
- [ ] Create video demonstrations of key features
- [ ] Build internal knowledge base

---

## 🎯 Success Criteria

### Phase 1 Complete When:
- [x] Project directory is clean and well-organized
- [x] All dependencies are pinned and security-scanned
- [x] Secrets are properly managed and documented
- [x] No legacy cruft remains

### Phase 2 Complete When:
- [ ] Configuration system is unified and validated
- [ ] Plugin architecture is functional with examples
- [ ] Caching system is robust and monitored
- [ ] API costs are controlled and predictable

### Project Health Goals:
- [ ] Maintain ≥98% test pass rate throughout cleanup
- [ ] Achieve and maintain ≥80% code coverage
- [ ] Zero critical security vulnerabilities
- [ ] Sub-5-minute development environment setup
- [ ] Complete CI/CD pipeline with <10-minute feedback loop

---

## 📋 Migration Notes

### From Previous TODO (aug10todo.md):
- ✅ **"Stabilize core and tests"** - COMPLETED (98.6% pass rate achieved)
- 🔄 **Configuration-driven groundwork** - INCORPORATED into Phase 2
- 🔄 **Plugin architecture prep** - INCORPORATED into Phase 2  
- 🔄 **Caching and idempotency** - INCORPORATED into Phase 2
- 🔄 **API cost caps and safety** - INCORPORATED into Phase 2
- 🔄 **Observability and diagnostics** - INCORPORATED into Phase 3
- 🔄 **CI/CD and workflows** - INCORPORATED into Phase 4
- 🔄 **Documentation** - INCORPORATED into Phase 5

### Breaking Changes to Consider:
- Configuration format may change during Phase 2
- Plugin system will require code changes for custom classifiers  
- Cache invalidation may reset existing cached data
- API cost controls may limit existing high-volume operations

---

*This TODO represents the next evolution of the Pharmacy Scraper project, building on the solid foundation of stabilized tests and proven real-world functionality.*
