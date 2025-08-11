#!/usr/bin/env bash
set -euo pipefail

# Simple developer security/dependency checks
# - pip-audit: scans installed packages for known vulnerabilities
# - detect-secrets: scans repo for potential committed secrets

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v pip-audit >/dev/null 2>&1; then
  echo "pip-audit not found. Install dev deps: pip install -r requirements-dev.txt" >&2
  exit 1
fi

if ! command -v detect-secrets >/dev/null 2>&1; then
  echo "detect-secrets not found. Install dev deps: pip install -r requirements-dev.txt" >&2
  exit 1
fi

echo "==> Running pip-audit (vulnerability scan)"
pip-audit || true

echo "\n==> Running detect-secrets (pre-commit style scan)"
# Use baseline if present; otherwise run a scan and summarize findings
BASELINE_FILE=".secrets.baseline"
if [[ -f "$BASELINE_FILE" ]]; then
  detect-secrets scan --baseline "$BASELINE_FILE" || true
else
  detect-secrets scan | tee .detect-secrets.report || true
  echo "\nNo baseline found. Consider creating one: detect-secrets scan > .secrets.baseline" >&2
fi

echo "\nSecurity/dependency checks completed."
