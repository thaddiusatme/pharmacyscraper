# Pharmacy Scraper Project Cleanup & Enhancement TODO
*Generated: 2025-08-11*

## 🎯 Current Status
- ✅ **Test Stabilization Complete:** 98.6% pass rate (340/345 tests), 73% coverage
- ✅ **Real-World Verification Complete:** Pipeline successfully processes real pharmacy data
- ✅ **Functionality Proven:** End-to-end execution with real APIs confirmed working
- ✅ **Fail-Fast Env Validation:** Centralized env validator integrated in production runner

---

## 🧹 Phase 1: Project Cleanup & Organization [IMMEDIATE]

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
- [ ] Remove unused root-level scripts and configs
- [ ] Archive old trial result directories in `data/`

### 📂 Directory Organization
- [x] Consolidate cache directories (`cache/`, `.api_cache/`, `data/cache/`)
- [x] Move scattered run scripts to `scripts/` directory
- [ ] Organize config files (keep production/dev/test structure)
- [x] Clean up `data/` directory (19+ subdirectories identified)
- [x] Archive or remove obsolete output directories

### 📌 Dependency Management
- [ ] Pin all dependency versions in `requirements.txt` (based on working 98.6% test suite)
- [x] Remove duplicate pytest entries (lines 2 and 30)
- [ ] Audit and remove unused dependencies
- [x] Create `requirements-dev.txt` for development-only deps
- [ ] Add dependency security scanning

---

## 🏗️ Phase 2: Architecture & Infrastructure [HIGH PRIORITY]

### 📋 Configuration System
- [ ] Define unified config schema (YAML/JSON) for search terms, regions, budgets
- [ ] Implement config validation with clear error messages (pydantic/voluptuous)
- [ ] Create example configs and migration guide from current setup
- [ ] Add environment-specific config inheritance

### 🔌 Plugin Architecture Foundation
- [ ] Design `BaseClassifierPlugin` interface with hooks for chain detection, compounding
- [ ] Implement plugin registry/loader system (entry points or module discovery)
- [ ] Create example plugin implementation
- [ ] Add plugin testing framework and validation

### 💾 Caching & Idempotency Enhancement
- [ ] Standardize deterministic cache keys across all modules
- [ ] Implement cache invalidation strategy and "force_reclassification" behavior
- [ ] Add cache hit/miss metrics and monitoring
- [ ] Create cache cleanup/maintenance utilities

### 💰 API Cost Management
- [ ] Implement per-search-term budget caps with fail-safe cutoffs
- [ ] Enhanced integration with existing API usage tracker
- [ ] Add cost prediction and budget planning tools
- [ ] Create cost reporting and analytics dashboard

---

## 📊 Phase 3: Observability & Quality [MEDIUM PRIORITY]

### 🔍 Enhanced Logging & Monitoring
- [ ] Implement structured logging (run_id, stage, source, cost, cache_source)
- [ ] Add comprehensive metrics: API calls, retries, cache hits, cost per term/region
- [ ] Create debug toggles and field redaction for sensitive data
- [ ] Build monitoring dashboards and alerting

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
- [ ] Project directory is clean and well-organized
- [ ] All dependencies are pinned and security-scanned
- [ ] Secrets are properly managed and documented
- [ ] No legacy cruft remains

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
