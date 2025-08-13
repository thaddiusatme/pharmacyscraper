# Project Manifest – Retail/Specialty Grocery Locations List Program

## 1. Vision
Deliver high-quality, normalized lists of retail locations (e.g., specialty/independent grocery) using a repeatable, compliant ingestion pipeline. The pipeline discovers candidate locations via web search/operators and store locators, ingests curated directories/CSVs, normalizes address/phone, deduplicates, and exports CSV/JSON for client consumption.

## 2. Objectives
- Discovery: Find retailer lists and store locator pages using targeted search operators and known directories.
- Ingestion: Scrape/store locator pages (robots-aware), import member directories/CSVs.
- Normalization: Produce segmented address fields and normalized phone numbers.
- Quality: Deduplicate, basic validation, and optional classification.
- Compliance: Respect robots.txt, rate limits, ToS; feature-flag advanced discovery.
- DX/Extensibility: Implement as plugins with test coverage and clear configs.

## 3. In-Scope / Out-of-Scope
- In-Scope:
  - Web search for “store locator” pages via API (SerpAPI, CSE, or Apify actor)
  - Store-locator scraping with JSON-LD and structured HTML fallbacks
  - Optional: Apify-backed locator scraping via actor (JSON-LD-first with HTML fallback)
  - Directory ingestion or CSV imports
  - Address/phone normalization; caching; export to CSV/JSON
- Out-of-Scope (initial phase):
  - Deep crawling beyond robots.txt allowances
  - Vendor outreach, email discovery, or CRM integrations

## 4. Key Deliverables
- D1: Configurable pipeline for business_type="specialty_grocery"
- D2: Source plugins
  - WebSearchSourcePlugin (find candidate locator URLs via operators)
  - StoreLocatorScraperPlugin (extract locations from locator pages)
  - MemberDirectorySourcePlugin (targeted directory ingestion) — optional
  - CSVSourcePlugin (existing) for curated inputs
- D3: Normalized CSV/JSON outputs with schema v2 fields
- D4: Docs: runbook, examples, and compliance notes
- D5: Backend selection + budgets & flags (Apify default), caching of discovery outputs (e.g., Apify run/dataset IDs)

## 5. Data Model (Schema v2 excerpts)
- Core fields (selected):
  - name, address_line1, address_line2, city, state, postal_code, country_iso2
  - phone_e164, phone_national
  - source_url, source_type (locator|directory|csv|websearch)
  - business_type (e.g., specialty_grocery)
- See `docs/SCHEMA.md` for full field list and gating.

## 6. Discovery Strategies (Client Operators)
Client-provided examples (to parametrize WebSearchSource):
- Retailer lists and store locators
  - `inurl:store-locator "Mediterranean market"`
  - `inurl:store-locator "gourmet market"`
  - `intitle:"store locations" "natural foods"`
  - `"independent grocery stores" list {STATE}`
  - `"specialty grocery" "store locations" {CITY}`
  

## 7. Architecture & Flow
- Orchestrator (plugin mode) with business_type="specialty_grocery"
- Sources (registry-driven):
  - WebSearchSourcePlugin or ApifyMapsSourcePlugin (Google Maps) or ApifyWebSearchSourcePlugin (SERP) → emits candidate places/URLs + metadata (via Apify actor when configured)
  - StoreLocatorScraperPlugin (in-process) or ApifyLocatorScraperPlugin (optional) → fetches allowed pages, extracts LocalBusiness items
  - MemberDirectorySourcePlugin → paginates or imports
  - CSVSourcePlugin → simple file import
- Processing:
  - Deduplication (simple name+address key, future: fuzzy)
  - Normalization (address via usaddress/libpostal; phone via phonenumbers)
- Output:
  - Typed cache keys (business_type-prefixed) with backward-compat fallback
  - Exports: CSV + JSON

### Apify integration (existing infrastructure)
- Actors
  - Search/Places: `apify.search_actor_id` (Google Maps Scraper actor; default: `nwua9Gu5YrADL7ZDj`). Default backend via `WEB_DISCOVERY_BACKEND=apify`.
    - Actor README: https://console.apify.com/actors/nwua9Gu5YrADL7ZDj/information/latest/readme
  - Optional locator crawler: `apify.crawler_actor_id` (e.g., Website Content Crawler or a custom locator extractor actor).
- Inputs (search actor)
  - Queries: operator-templated strings expanded by {STATE}/{CITY}.
  - Limits: `maxItems`, `maxPagesPerQuery` (if supported), domain allow/deny lists.
  - Regional hints: country/locale parameters (if supported by actor).
- Outputs
  - Dataset items containing candidate places and/or URLs (e.g., name, formatted address, lat/lng, phone, website, categories, placeId, url). We retain fields needed for normalization and provenance.
  - Naming: dataset/run tags include `business_type`, `region`, and a date stamp.
- Control flow
  - Plugin starts an Apify run (or reuses a cached run) and waits for finish (poll or `waitForFinish`).
  - Retrieve dataset items via API; emit into pipeline as discovery results.
- Caching
  - Cache `(business_type, operator_template, region)` → `{ runId, datasetId, fetched_at }`.
  - Freshness window (default 7 days). Skip new run if cache is fresh.
- Budgets & rate
  - Enforce `apify.max_runs_per_day`, `apify.max_concurrency`, and per-run budget/time limits from config.
  - Fail fast if limits are hit; partial results are still cached.
- Compliance
  - Respect robots.txt and ToS; maintain a domain allowlist; identifiable user-agent.
  - Shut off discovery via `WEB_DISCOVERY_ENABLED=0` for code-only/test runs.

## 8. Milestones & Timeline
| Week | Milestone | Acceptance |
|------|-----------|------------|
| 1 | Scaffold ApifyMapsSourcePlugin (default backend) | Finds ≥20 relevant locator URLs OR ≥20 relevant Google Maps places for 2 states |
| 2 | StoreLocatorScraperPlugin (JSON-LD + HTML patterns) or ApifyLocatorScraperPlugin | Extracts ≥100 locations from 3 retailer locators; unit tests pass |
| 2 | Backend selection + flags + budgets | WEB_DISCOVERY_BACKEND=apify; budget caps set; discovery output caching enabled |
| 3 | Directory ingestion (opt.) + CSV import path | Ingests sample from a public directory export or CSV |
| 3 | Dedup + normalization wired | Outputs segmented address + phone; cache keys typed |
| 4 | Docs + runbook + examples | End-to-end run instructions; sample configs; compliance notes |

## 9. Acceptance Criteria
- End-to-end run produces CSV/JSON of ≥200 locations across at least 2 regions
- Address fields segmented and phone normalized for ≥95% of rows
- Caching prevents duplicate fetch for repeated runs
- Tests: unit + integration added for new plugins; suite remains green
- Deterministic tests use recorded fixtures (including Apify dataset samples); live runs gated behind flags and budgets

## 10. Compliance & Risk Management
- Robots.txt honored; politeness delays; limited depth; domain allowlists
- Feature flags to enable/disable web discovery; rate limiting; user-agent identification
- Avoid scraping login-protected or restricted content; respect ToS
- Privacy: Do not collect PII beyond business contact details; logs avoid full phone numbers by default
- Cache Apify run/dataset IDs and enforce actor budget limits; domain allowlists and per-domain politeness

## 11. Feature Flags & Config
- `WEB_DISCOVERY_ENABLED=0|1` (new)
- `WEB_DISCOVERY_BACKEND=apify|cse|serpapi` (new)
- `INTERNATIONAL_ENABLED=0|1` (existing)
- Budget limits for any paid search API
- Query templates for operators and {STATE}/{CITY} expansion
- Environment: `APIFY_TOKEN`
- Apify config: `apify.search_actor_id`, `apify.crawler_actor_id` (optional), concurrency, per-run and per-day budgets, caching of run/dataset IDs
  - Defaults: `apify.search_actor_id = nwua9Gu5YrADL7ZDj`

## 12. Testing Strategy
- Unit tests for parsing JSON-LD and common locator HTML patterns
- Mocked web fixtures for deterministic tests
- Integration tests: WebSearchSource → StoreLocatorScraper → normalization → export
- Recorded Apify dataset samples used in tests for backend determinism

## 13. Runbook (Example)
```bash
# Backend/env (if using Apify)
export WEB_DISCOVERY_BACKEND=apify
export APIFY_TOKEN=xxxxxx  # set your Apify token

# Example: specialty grocery in CA using Google Maps discovery via Apify
python -m pharmacy_scraper.run_plugin_pipeline \
  --business-type specialty_grocery \
  --config config/locations_maps_discovery.json \
  --output ./output/locations

# Using a curated CSV
python -m pharmacy_scraper.run_plugin_pipeline \
  --business-type specialty_grocery \
  --config config/locations_from_csv.json \
  --output ./output/locations
```

## 14. Open Questions
- Apify backend confirmed. Any locator crawler actor to add? Daily budget caps?
- Regional scope (states/cities) for first delivery? Target retailer examples?
- Directory access/terms for specialtyfood.com and others?

## 15. Definition of Done (DoD)
1. Pipeline produces validated location lists meeting acceptance criteria
2. Documentation (manifest, runbook, configs) reviewed with client
3. Tests pass in CI; coverage maintained
4. Artifacts delivered (CSV/JSON + metadata)

## 16. Decision Gates & Go/No-Go
- Gate A (end of Week 1): Discovery backend sanity check
  - Criteria: ≥20 relevant locator URLs for 2 states via operators; budget burn within cap; robots.txt logs clean
  - Decision: proceed with Apify as default (or switch to CSE/SerpAPI) and finalize operator templates
- Gate B (mid Week 2): Locator extraction viability
  - Criteria: ≥100 locations extracted across ≥3 retailer locators with JSON-LD-first + HTML fallback
  - Decision: scale states/cities; prioritize pattern gaps with fixtures
- Gate C (Week 3 end): Data quality readiness
  - Criteria: ≥95% rows with segmented address + normalized phone; duplicate rate ≤3%
  - Decision: proceed to broader run and client review

## 17. Budget & Rate Guardrails
- Backend: `WEB_DISCOVERY_BACKEND=apify` (default)
- Discovery limits (initial defaults; configurable):
  - `discovery.max_regions_per_run`: 5
  - `discovery.max_items_per_region`: 50
  - `apify.max_runs_per_day`: 10; `apify.max_concurrency`: 5
  - Per-run cost budget: $X (set with client); hard-stop on exceed
- Caching:
  - Cache Apify run/dataset IDs per (business_type, query, region)
  - Reuse cached datasets when within freshness window (e.g., 7 days)

## 18. QA Sampling & Data Quality
- Sampling: Review 5–10% of rows per region manually for name/address/phone accuracy and valid source_url
- Dedup rules:
  - Primary key: normalized(name) + address_line1 + postal_code
  - Secondary fuzzy check (future): token-sort ratio on name + city
- Coverage metrics:
  - JSON-LD vs HTML extraction ratio; parse error rate; normalization success (%)
  - Report per-domain/domain-group to guide parser improvements

## 19. Assumptions & Dependencies
- APIFY_TOKEN available and actor choices confirmed (search actor, optional crawler)
- Operator templates provided/approved; initial region list (states/cities) scoped
- Domain allowlist maintained; robots.txt respected by design; legal review as needed
