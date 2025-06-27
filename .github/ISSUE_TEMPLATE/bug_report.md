---
name: 🐛 Bug Report
about: Report an issue with the Pharmacy Scraper
title: '[BUG] Brief description of the issue'
labels: 'bug'
---

## 🐞 Bug Description
A clear and concise description of what the bug is.

## 🔍 Affected Components
- [ ] Data Collection (Apify)
- [ ] Classification (Perplexity API)
- [ ] Deduplication
- [ ] Self-Healing
- [ ] Verification System
- [ ] Documentation
- [ ] Other (please specify)

## 🐜 Steps to Reproduce
1. Environment setup:
   - OS: [e.g., macOS 14.0, Ubuntu 22.04]
   - Python Version: [e.g., 3.9.6]
   - Branch: [e.g., main, feature/independent-pharmacy-filter]

2. Steps to reproduce:
   ```bash
   # Example commands that trigger the bug
   python scripts/run_scraper.py --config config/production.json
   ```

## 🎯 Expected vs Actual Behavior
**Expected:**
- What should happen

**Actual:**
- What actually happens
- Include any error messages or stack traces

## 📊 Impact Assessment
- [ ] Data loss/corruption
- [ ] Incorrect classification
- [ ] Performance degradation
- [ ] API cost implications
- [ ] Security concern

## 📝 Additional Context
- Screenshots (if applicable)
- Related issues/PRs
- Environment details:
  - Apify token configured: [Yes/No]
  - Google Places API key: [Yes/No]
  - Perplexity API key: [Yes/No]

## ✅ To Reproduce (for developers)
```python
# Minimal code to reproduce the issue
from pharmacy_scraper import PharmacyScraper

# Add test code here
```

## 📋 Additional Information
- First occurrence: [Date/Time]
- Frequency: [Always/Intermittent/Specific conditions]
- Workarounds (if any):

## 🏷 Labels
- bug
- needs-triage

## 🙋‍♂️ Assignee
@[assignee]
