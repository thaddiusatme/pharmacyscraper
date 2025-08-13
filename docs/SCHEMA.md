# Data Schema — Version 2

Schema version: `2`
Status: Active (introduced by Project Crosswalk)

This document defines the fields and conventions for CSV and JSON outputs produced by the pipeline as of Schema Version 2. v2 adds normalized address fields and contact enrichment fields while keeping backwards compatibility for existing CSV consumers (new columns appended) and preserving snake_case for JSON.

Note: Implementation is landing iteratively via TDD. As of Iteration 7, all planned v2 serialization fields are present and the orchestrator wires normalization prior to serialization. Address normalization uses usaddress for US and libpostal for international when `INTERNATIONAL_ENABLED=1`; phone normalization uses phonenumbers with fallbacks. See "Current Implementation Status" below.

## 1. Versioning and Compatibility
- `schema_version`: 2 for new runs. It SHOULD be present in JSON outputs and run metadata. For CSV exports, versioning is tracked in run metadata/file naming; CSV contains the appended columns for v2.
- Backward compatibility: All v1 columns remain unchanged. v2 columns are appended to CSV. JSON simply includes additional fields as described below.
- Backfill: By default, normalization applies to new runs only. If a backfill is executed, it writes to a separate folder labeled with `schema_version=2`.

## 2. Field Catalog (new/changed in v2)
The following fields are introduced or standardized in v2. Types refer to JSON representations; CSV stores strings unless noted.

Address normalization (US-first; international optional):
- `address_line1` (string): Primary street line including number and street name.
  - Example: "123 Main St"
- `address_line2` (string|null): Secondary unit info (suite, apt, PO Box). Empty if not applicable.
  - Example: "Suite 400"
- `city` (string): City or locality.
  - Example: "Austin"
- `state` (string): State/region for US (two-letter, when applicable).
  - Example: "TX"
- `postal_code` (string): Postal/ZIP code, including suffix if available.
  - Example: "78701" or "78701-1234"
- `country_iso2` (string): ISO 3166-1 alpha-2 country code when known.
  - Example: "US"
- `country_code` (string, optional): Present when `INTERNATIONAL_ENABLED=1` and available. ISO 3166-1 alpha-2; may duplicate `country_iso2` depending on source.
  - Example: "US"

Phone normalization:
- `phone_e164` (string|null): E.164 formatted phone number (e.g., +15551234567) when available.
  - Example: "+15125551234"
- `phone_national` (string|null): National format phone number string when available.
  - Example: "(512) 555-1234"

Contact enrichment:
- `contact_name` (string|null): Canonical contact owner name when available.
  - Primary source: NPI Authorized Official (when applicable).
- `contact_role` (string|null): Role/title of contact (e.g., Authorized Official).
- `contact_source` (string|null): Origin of contact fields.
  - Values: `npi_authorized_official` | `website` | `api` | null
- `contact_email` (string|null): Contact email when confidently sourced from APIs (Phase 1). If optional website discovery is enabled, may be populated from website.
- `contact_email_source` (string|null): Source of `contact_email`.
  - Values: `api` | `scrape` | null

Notes:
- JSON remains snake_case.
- CSV includes the above fields as appended columns.

## 3. Source Precedence and Policies
- Phone: Google Places > NPI > Website.
- Contact name/role: NPI Authorized Official (`contact_source = npi_authorized_official`) > Website/API generic contact > null.
- Email: API-only in Phase 1. If website discovery is enabled, tag `contact_email_source` accordingly and only store RFC5322-valid addresses.

## 4. Feature Flags & Defaults
- `EMAIL_DISCOVERY_ENABLED` (default 0)
  - When enabled: respect robots.txt, use a domain allowlist, crawl up to 2 pages, tag `contact_email_source = scrape` for discovered emails.
- `INTERNATIONAL_ENABLED` (default 0)
  - When enabled: enable international parsing with libpostal, populate `country_code` when available, and gate non-US logic accordingly.

## 5. Privacy & Logging
- PII redaction is ON by default.
- Logs and metrics MUST NOT contain raw emails or full phone numbers.
- Metrics include only counts/durations and SHOULD NOT use PII-bearing label values.

## 6. Dependencies (Docs)
- Approved now: `phonenumbers` (phone normalization), `usaddress` (US address parsing).
- Gated: `libpostal` (used only when `INTERNATIONAL_ENABLED=1`).
- All dependencies are optional. When unavailable, the pipeline falls back to heuristics while preserving output schema.

## 7. CSV and JSON Examples

CSV columns (excerpt; v1 columns omitted for brevity; v2 appended):
```
...,address_line1,address_line2,city,state,postal_code,country_iso2,phone_e164,phone_national,contact_name,contact_email,contact_role,contact_source,contact_email_source[,country_code]
```
Notes:
- `contact_source` and `contact_email_source` are always included as columns (values may be null).
- `country_code` column is included only when `INTERNATIONAL_ENABLED=1`.

JSON example (excerpt):
```json
{
  "schema_version": 2,
  "name": "Example Pharmacy",
  "address_line1": "123 Main St",
  "address_line2": "Suite 400",
  "city": "Austin",
  "state": "TX",
  "postal_code": "78701-1234",
  "country_iso2": "US",
  "phone_e164": "+15125551234",
  "phone_national": "(512) 555-1234",
  "contact_name": "Jane Doe",
  "contact_role": "Authorized Official",
  "contact_source": "npi_authorized_official",
  "contact_email": "info@examplepharmacy.com",
  "contact_email_source": "api"
}
```

## 9. Current Implementation Status (Iteration 7)
- Appended to CSV and present in JSON: `address_line1`, `address_line2`, `city`, `state`, `postal_code`, `country_iso2`, `phone_e164`, `phone_national`, `contact_name`, `contact_email`, `contact_role`, `contact_source`, `contact_email_source`.
- `country_code` is present in JSON and appended to CSV only when `INTERNATIONAL_ENABLED=1`.
- Normalization is integrated into the orchestrator prior to serialization:
  - US addresses parsed via `usaddress` when available; otherwise heuristic fallback.
  - International addresses parsed via `libpostal` when enabled and available; otherwise heuristic fallback.
  - Phones formatted via `phonenumbers` (E.164 and national) with graceful fallback.
- Next iterations: NPI Authorized Official contact enrichment, email policy gating/validation, privacy redaction in logs/metrics.

## 8. Changelog (v1 → v2)
- Added normalized address fields: line1, line2, city, state, postal_code, country_iso2.
- Added contact enrichment fields: contact_name, contact_role, contact_source, contact_email, contact_email_source.
- Added phone normalization fields: phone_e164, phone_national.
- Introduced feature flags: EMAIL_DISCOVERY_ENABLED, INTERNATIONAL_ENABLED.
- Established privacy/metrics defaults (no raw emails or full phones in logs/metrics).
- Schema version bumped to 2; CSV columns appended; JSON remains snake_case.
