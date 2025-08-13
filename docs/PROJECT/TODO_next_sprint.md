# TODO – Sprint X (Search-Agnostic Refactor)

## High-priority
- [x] **Package alias / shim** – Create `business_scraper` namespace and re-export existing code; maintain `pharmacy_scraper` for compatibility.
- [x] **Config schema update** – Add `business_type`, `search_terms` fields; migrate loaders; update docs.
- [ ] **DomainClassifier interface** – Define in `classification/domain.py`; implement `PharmacyClassifier` plug-in.
- [ ] **Rule externalisation** – Move chain keywords → `rules/pharmacy.yml`; build loader.
- [ ] **Prompt template system** – Jinja2 templates in `prompts/<domain>.j2`; update `PerplexityClient` to load.

## Medium
- [ ] Cache-key migration script with backward lookup.
- [ ] Update orchestrator to accept `business_type` and use plug-in registry.
- [ ] Add example second domain (`vet_clinic`) + tests.
- [ ] Update READMEs, module docs, diagrams.

## Crosswalk: Address & Contact Normalization (Related)
- See: docs/PROJECT/manifest_crosswalk.md
- See also: docs/SCHEMA.md (Schema v2)
- [x] Bump SCHEMA_VERSION=2; append CSV columns; JSON remains snake_case; document in SCHEMA.md
- [x] Add normalized address/contact v2 fields to outputs (CSV tail + JSON): `address_line1`, `address_line2`, `city`, `state`, `postal_code`, `country_iso2`, `phone_e164`, `phone_national`, `contact_name`, `contact_email`, `contact_role`, `contact_source`, `contact_email_source`; gate `country_code` by `INTERNATIONAL_ENABLED`.
- [x] TDD (Red): add failing unit tests for address and phone normalization
- [x] TDD (Green minimal): add `pharmacy_scraper/normalization/address.py` and `phone.py` with minimal logic to satisfy tests
- [x] TDD (Refactor/Integrate): wire normalization into pipeline (populate fields in orchestrator before serialization); preserve feature flag gating and BC
- [x] Implement normalization engine (US: `usaddress`; International: `libpostal` gated by `INTERNATIONAL_ENABLED=1`) with graceful fallbacks
- [ ] Phone normalization via `phonenumbers` (E.164 and national) — enhance error handling where needed
- [ ] Contact enrichment: use NPI Authorized Official as canonical `contact_name`/`contact_role`; set `contact_source="npi_authorized_official"` when used
- [ ] Email policy Phase 1: API-only; optional website discovery behind `EMAIL_DISCOVERY_ENABLED` with robots.txt, domain allowlist, ≤2 pages; RFC5322 validation; tag `contact_email_source={api|scrape}`
- [ ] Feature flags: add `EMAIL_DISCOVERY_ENABLED=0`, `INTERNATIONAL_ENABLED=0` to config/env; document defaults
- [ ] Dependency gating: approve `phonenumbers`, `usaddress`; guard `libpostal` behind `INTERNATIONAL_ENABLED`
- [ ] PII-safe logs/metrics: redact by default; no raw emails or full phone numbers; metrics only counts/durations; track field fill-rate

Notes (Iteration 7):
- Orchestrator now calls address/phone normalization prior to JSON/CSV serialization and respects `INTERNATIONAL_ENABLED` for `country_code`.
- Docs updated: see `docs/SCHEMA.md` for normalization details and dependencies.

## Low
- [ ] Deprecation warning in shim package.
- [ ] Coverage report target >=70 % after refactor.
- [ ] Create CHANGELOG entry.

---
Run `pytest -q` after each major step and keep CI green.
