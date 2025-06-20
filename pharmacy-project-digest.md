# Independent Pharmacy Verification Project
## Complete Project Digest

### 📋 Project Overview
**Client**: Pharmacy mailing list verification  
**Budget**: $350  
**Timeline**: 3-4 days  
**Deliverable**: Verified list of 1,250 independent pharmacies (25 per state) with mailing addresses

---

## TDD Implementation Status

### ✅ Completed with TDD
1. **Perplexity API Client**
   - Implemented with retry logic and rate limiting
   - Comprehensive test coverage for success/error cases
   - Mocked API responses for reliable testing

2. **Caching Layer**
   - TTL and size-based eviction
   - File-based persistence
   - Tested cache hit/miss behavior
   - Verified cache persistence between runs

3. **Classifier Integration**
   - Batch processing with progress tracking
   - Rule-based fallback classification
   - Test coverage for all major scenarios
   - Error handling and logging

### 🔄 In Progress
- Full test suite integration
- Documentation updates
- Performance optimization

## Phase 1: Data Collection (Day 1-2)

### 🎯 Primary Approach: Apify Google Maps Scraper
**Cost**: ~$5 | **Time**: 45 minutes | **Success Rate**: 95%

1. **Setup Apify** (15 minutes)
   - Create account at https://apify.com
   - Navigate to Google Maps Scraper
   - Add payment method (pay only for results)

2. **Configure Searches**
   - Search queries format: `independent pharmacy [City] [State]`
   - 3-5 major cities per state
   - Max 30 results per search
   - Filter out chains: CVS, Walgreens, Rite Aid, Walmart, Target

3. **Export Results**
   - Download as CSV
   - Expected: 1,000-1,500 pharmacies
   - Fields: Name, Address, City, State, ZIP, Phone, Status

### 🔄 Backup Approach: Free APIs
**If Apify unavailable or over budget**

#### Option A: NPI Registry (Free)
```python
# No API key needed!
url = "https://npiregistry.cms.hhs.gov/api/"
params = {
    'version': '2.1',
    'state': 'TX',
    'taxonomy_description': 'pharmacy',
    'limit': 200
}
```

#### Option B: Yelp API (Free tier)
- 5,000 calls/day free
- Get API key: https://www.yelp.com/developers
- Use provided Yelp script
- Excellent for finding independents

#### Option C: CMS Medicare Data
- Download from: https://data.cms.gov/provider-data/topics/pharmacies
- Filter for empty chain_code
- Instant 20,000+ pharmacies

---

## Phase 2: Data Verification (Day 2-3)

### ✅ Verification Script (Google Places API)
**Cost**: ~$21 | **Time**: 2-3 hours for 1,250 locations

```python
# Key components of verification script:
1. Search by business name + address
2. Check business_status field
3. Verify address matches
4. Output confidence scores

# Expected results:
- OPEN (High confidence)
- CLOSED (Permanently closed)
- NOT_FOUND (Needs manual check)
- UNCERTAIN (Low confidence)
```

### 📊 Verification Process
1. **Batch Processing**
   - Process in groups of 100-200
   - Rate limit: 0.5 seconds between calls
   - Save progress regularly

2. **Quality Checks**
   - Spot check 5-10% manually
   - Review all "UNCERTAIN" results
   - Verify states with <25 pharmacies

3. **Final Output Format**
   ```csv
   Name, Street Address, City, State, ZIP, Status, Confidence, Notes
   ```

---

## 🚀 Recommended Fast Track (Complete in 1 Day)

### Morning (2 hours)
1. **Apify Setup & Run**
   - Configure for all 50 states
   - Run scraper (~45 min)
   - Download results
   - Quick data cleaning

### Afternoon (3 hours)
2. **Verification**
   - Run Google Places verification script
   - Process results
   - Fill any gaps with NPI Registry
   - Final QA check

### Evening (1 hour)
3. **Delivery**
   - Format final spreadsheet
   - Create summary report
   - Package for client

---

## 💰 Budget Breakdown

| Item | Cost | Purpose |
|------|------|---------|
| Apify Scraping | $5 | Initial data collection |
| Google Places API | $21 | Verification of 1,250 locations |
| **Total Costs** | **$26** | |
| **Your Profit** | **$324** | ~$80/hour for 4 hours work |

---

## 📁 Final Deliverables

1. **Master Spreadsheet** (independent_pharmacies_verified.csv)
   - All 1,250 pharmacies
   - Complete mailing addresses
   - Verification status
   - Confidence scores

2. **Summary Report**
   - Pharmacies per state breakdown
   - Verification statistics
   - Any states needing attention

3. **Raw Data Files**
   - Original Apify export
   - Verification results
   - Error log (if any)

---

## ⚡ Quick Commands Reference

```bash
# Install required packages
pip install pandas requests googlemaps

# Test scripts
python test_single_pharmacy.py

# Run verification
python verify_pharmacies.py input.csv output.csv

# Generate report
python create_summary.py output.csv
```

---

## 🎯 Success Metrics

- ✅ 1,250 total pharmacies (25 per state)
- ✅ 95%+ verification confidence
- ✅ All currently operational
- ✅ Complete mailing addresses
- ✅ Delivered within budget & timeline

---

## 🔧 Troubleshooting

**If short on pharmacies in some states:**
1. Use NPI Registry for those states
2. Search additional cities
3. Try "compounding pharmacy" searches
4. Check state pharmacy board websites

**If verification shows many closures:**
1. Re-run Apify with "currently open" filter
2. Cross-reference with Yelp
3. Use backup sources for replacements

**If over budget:**
1. Use free NPI Registry exclusively
2. Verify only uncertain results
3. Sample verification (every 3rd pharmacy)

---

## 📞 Quick Support Resources

- **Apify Support**: support@apify.com
- **Google Cloud Console**: https://console.cloud.google.com
- **NPI Registry**: https://npiregistry.cms.hhs.gov
- **Yelp Developers**: https://www.yelp.com/developers

---

## ✨ Project Success Tips

1. **Start with Apify** - It's the fastest, most reliable method
2. **Verify in batches** - Easier to troubleshoot issues
3. **Keep raw data** - Client may want additional fields later
4. **Document everything** - For potential follow-up projects
5. **Under-promise, over-deliver** - Aim for 30 per state, deliver 25

---

### Next Steps
1. Set up Apify account
2. Run test with 2-3 states
3. Verify test results work with verification script
4. Run full collection & verification
5. Deliver to client

**Estimated Total Time**: 4-6 hours of actual work
**Profit Margin**: 92% ($324/$350)