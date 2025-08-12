# Project Crosswalk — Search-Agnostic Refactor + Address/Contact Normalization

Branch: `feature/search-agnostic-refactor`
Window: <!-- fill dates -->

## 1. Vision
Evolve the pharmacy-focused pipeline into a domain-agnostic business data framework while normalizing addresses into structured fields and enriching outputs with phone, contact owner, and email where available.

## 2. Objectives
1) Make collection/classification/verification domain-agnostic with plug-in points.
2) Normalize output addresses into: `address_line1`, `address_line2`, `city`, `state`, `postal_code`, `country_iso2`.
3) Add contact enrichment: `phone_e164`, `phone_national`, `contact_name` (owner/authorized official), `contact_email` (if available).
4) Preserve backward compatibility (pharmacy shim), budget controls, and test stability.

## 3. Deliverables and Acceptance Criteria
D1. Domain-agnostic core + `business_scraper` namespace
- AC: Pipeline runs under both `business_scraper` and `pharmacy_scraper` shim; all tests pass.

D2. Config schema update (`business_type`, `search_terms`)
- AC: Runs a non-pharmacy domain in test mode with docs/examples.

D3. `DomainClassifier` interface + `PharmacyClassifier` plug-in
- AC: Unit tests for at least two domains (pharmacy + one example), rules externalized.

D4. Prompt templating (`prompts/<domain>.j2`)
- AC: Correct domain prompt selected at runtime; tested.

D5. Cache-key migration (include `business_type`)
- AC: Backward lookup for pharmacy; no cache collisions across domains.

D6. Address normalization module
- AC: ≥95% US addresses successfully segmented using Google Places `address_components` (primary) and library fallback; validated format for international where possible.
- AC: `address_line1` contains street + number; `address_line2` contains unit/suite/PO Box.
- AC: Postal codes normalized; country ISO2 stored.

D7. Contact enrichment
- AC: Phone normalized to E.164 and national formats when available (Google Places and/or NPI).
- AC: `contact_name` and `contact_role` populated from NPI Authorized Official when available with `contact_source="npi_authorized_official"`; otherwise fallback to website/API generic contact; null-safe.
- AC: Phase 1: `contact_email` restricted to API-only sources. If `EMAIL_DISCOVERY_ENABLED=1` later, scraping respects robots.txt, domain allowlist, max 2 pages, and tags `email.source={api|scrape}`; emails validated RFC5322.

D8. PII-safe logging + observability
- AC: Redaction in logs for phone/email; logs and metrics contain no raw emails or full phone numbers; metrics only include counts/durations with no PII-bearing labels.
- AC: Field-level fill-rate metrics for address/contact fields.

D9. Docs + migration guide
- AC: README, config docs, examples updated; SCHEMA.md authored with `SCHEMA_VERSION=2`; CHANGELOG entry; adoption guide for new fields and versioning.

## 4. Workstreams and Tasks

A) Search-Agnostic Refactor (tracked in existing docs)
- A1. Namespace: `business_scraper` + `pharmacy_scraper` shim; deprecation notice
- A2. Config schema + orchestrator updates
- A3. `DomainClassifier` and first plug-in; rules externalization
- A4. Prompt templating + client integration
- A5. Cache-key migration script
- A6. Example second domain (`vet_clinic`) + tests
- A7. Docs/diagrams/READMEs

B) Address & Contact Normalization (new)
- B1. Data model updates:
  - Add fields: `address_line1`, `address_line2`, `city`, `state`, `postal_code`, `country_iso2`, `phone_e164`, `phone_national`, `contact_name`, `contact_email`, `contact_role` (optional)
  - Output scope: include fields in both CSV and JSON. CSV columns are appended (backward-compatible); JSON remains snake_case.
  - Bump `SCHEMA_VERSION=2` and document in `docs/SCHEMA.md`.
  - Update CSV/JSON writers and any downstream schemas.
- B2. Normalization engine:
  - Primary: Google Places `address_components`.
  - Fallback: `usaddress` for US. International parsing is optional and off by default; when `INTERNATIONAL_ENABLED=1`, pull in `libpostal`, add `country_code`, and gate non-US logic behind this flag.
  - Validation: unit tests with diverse address fixtures.
- B3. Contact extraction:
  - From Google Places: phone fields; email rarely available.
  - From NPI (already planned for verification fallback): Authorized Official name/role and phone; set `contact_source="npi_authorized_official"` when used.
  - Phase 1: Email restricted to API-only sources. Optional (guarded by `EMAIL_DISCOVERY_ENABLED`): lightweight site check for `mailto:`/contact page, respecting robots.txt and a domain allowlist, crawling ≤2 pages; tag `email.source` accordingly.
- B4. Standardization:
  - Phone: E.164 via `phonenumbers`.
  - Email: RFC5322 validation; no scraping without explicit config.
- B5. Source precedence and null-safety:
  - Phone: GPlaces > NPI > Website.
  - Contact name/role: NPI Authorized Official (`contact_source="npi_authorized_official"`) > Website/API generic contact > null.
  - Email: API-only by default; if website enabled, tag `email.source={api|scrape}` and store only validated addresses.
- B6. Observability & Privacy:
  - Redact PII in logs by default.
  - Logs and metrics contain no raw emails or full phone numbers; metrics only counts/durations; no PII in labels.
  - Metrics: fill-rate for each normalized field; error counts; parsing fallbacks.
- B7. QA & Backfill:
  - Add test fixtures and integration tests across sources.
  - New runs only for normalization by default. Optional backfill job writes to a new folder with `schema_version=2`.

## 5. Milestones (suggested)
Week 1: A1–A2; B1 schema + CSV updates
Week 2: A3–A4; B2 normalization engine + tests
Week 3: A5–A6; B3–B4 contact extraction and standardization
Week 4: A7; B5–B7 docs, metrics, privacy, and backfill

## 5.1 Feature Flags & Defaults
- `EMAIL_DISCOVERY_ENABLED` (default 0): When enabled, respect robots.txt, use a domain allowlist, crawl up to 2 pages, and tag `email.source={api|scrape}`.
- `INTERNATIONAL_ENABLED` (default 0): When enabled, use `libpostal` for non-US parsing, add `country_code`, and gate international logic accordingly.

## 6. Risks and Mitigations
- PII handling: Default redaction; explicit opt-in for email scraping.
- Address edge cases: Use multi-source approach and fallbacks; track fill-rate and error metrics.
- API cost drift: Budget caps per domain/term; cache-first behavior.
- Test flakiness: Expand mocks and fixtures; keep CI strict.

## 7. Definition of Done
- Two domains supported end-to-end.
- Address fields ≥95% parsed for US datasets; zero hard failures on international.
- Phone normalized to E.164 when present; contact fields populated per precedence.
- `SCHEMA_VERSION=2` implemented and documented in `docs/SCHEMA.md`; CSV columns appended; JSON snake_case maintained.
- Feature flags present with defaults off: `EMAIL_DISCOVERY_ENABLED=0`, `INTERNATIONAL_ENABLED=0`.
- All tests pass; docs updated; CHANGELOG created.
