# Sprint X Manifest – “Search-Term Agnostic Refactor”

**Branch:** `feature/search-agnostic-refactor`
**Sprint window:** <!-- fill dates -->

Related documents
- Project Crosswalk (Address & Contact Normalization): docs/PROJECT/manifest_crosswalk.md
- Schema v2 specification: docs/SCHEMA.md

---
## 1 Vision
Transform the current pharmacy-centric pipeline into a generic **Business-Data Scraper** framework that accepts any list of search terms (e.g., *“veterinary clinic”, “car wash”, “hardware store”*) while preserving budgeting, caching, and high test coverage.

## 2 Objectives
1. Abstract collection, classification, and verification layers so they are *domain-agnostic*.
2. Provide plug-in points for domain-specific heuristics (rule lists, prompts, attributes).
3. Maintain backward-compatibility by shipping a thin `pharmacy_scraper` shim.
4. Keep API-budget enforcement, caching, and CI test suite green throughout.

## 3 Key Deliverables
| # | Deliverable | Acceptance criteria |
|---|-------------|----------------------|
| D1 | Generic package namespace `business_scraper` (+ alias) | All imports pass and tests run via both namespaces. |
| D2 | Config schema supports `business_type` + `search_terms` | Pipeline runs successfully with a non-pharmacy term in test mode. |
| D3 | `DomainClassifier` strategy interface + first plug-in for pharmacy | Unit tests prove rule + LLM flow for at least two domains. |
| D4 | Prompt templating system (`prompts/<domain>.j2`) | Correct domain-specific prompt chosen at runtime. |
| D5 | Cache-key update includes `business_type` | Old keys still recognised for pharmacy; new keys for other domains. |
| D6 | Updated docs, README, examples, GitHub Actions | Documentation builds without pharmacy-only language. |

## 4 Milestones & Timeline
| Week | Milestone | Owner | Status |
|------|-----------|-------|--------|
| 1 | Repo scaffolding & alias/shim | | ☐ |
| 1 | Config schema & orchestrator param | | ☐ |
| 2 | DomainClassifier interface & pharmacy plug-in | | ☐ |
| 2 | Rule/keyword externalisation | | ☐ |
| 3 | Prompt templating & Perplexity integration | | ☐ |
| 3 | Cache-key migration script | | ☐ |
| 4 | Second sample domain (vet clinic) + tests | | ☐ |
| 4 | Documentation sweep | | ☐ |

## 5 Risks & Mitigations
- **Breaking users’ imports** → Provide `pharmacy_scraper` shim and deprecation notice.
- **API cost spikes** → Keep budget tracker; add per-term caps.
- **Prompt quality drop** → Iterate on prompt templates & include few-shot examples.
- **Test flakiness** → CI enforces coverage; keep mocks for external APIs.

## 6 Definition of Done
1. Pipeline can run against at least two distinct search terms without code changes.
2. All tests pass (`pytest -q`) with ≥ 70 % overall coverage.
3. README and module docs updated; CHANGELOG entry added.
4. PR merged to `main` after peer review and CI success.
