# TODO – Sprint X (Search-Agnostic Refactor)

## High-priority
- [ ] **Package alias / shim** – Create `business_scraper` namespace and re-export existing code; maintain `pharmacy_scraper` for compatibility.
- [ ] **Config schema update** – Add `business_type`, `search_terms` fields; migrate loaders; update docs.
- [ ] **DomainClassifier interface** – Define in `classification/domain.py`; implement `PharmacyClassifier` plug-in.
- [ ] **Rule externalisation** – Move chain keywords → `rules/pharmacy.yml`; build loader.
- [ ] **Prompt template system** – Jinja2 templates in `prompts/<domain>.j2`; update `PerplexityClient` to load.

## Medium
- [ ] Cache-key migration script with backward lookup.
- [ ] Update orchestrator to accept `business_type` and use plug-in registry.
- [ ] Add example second domain (`vet_clinic`) + tests.
- [ ] Update READMEs, module docs, diagrams.

## Low
- [ ] Deprecation warning in shim package.
- [ ] Coverage report target >=70 % after refactor.
- [ ] Create CHANGELOG entry.

---
Run `pytest -q` after each major step and keep CI green.
