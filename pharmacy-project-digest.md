# Independent Pharmacy Verification Project
## Complete Project Digest

### 📋 Project Overview
**Client**: Pharmacy mailing list verification  
**Budget**: $350  
**Timeline**: 3-4 days  
**Deliverable**: Verified list of 1,250 independent pharmacies (25 per state) with mailing addresses

---

## 🚀 Latest Trial Run Results (June 2024)

### ✅ Successfully Fixed and Optimized
**Issue Resolved**: Apify actor input validation errors causing 400 Bad Request failures
**Root Cause**: Invalid `allPlacesNoSearchAction: "false"` field value
**Solution**: Changed to empty string `""` to match actor requirements

### 📊 Trial Results Summary
- **Total Collected**: 39 independent pharmacies
- **California**: 19 pharmacies (Los Angeles: 10, San Francisco: 10) 
- **Texas**: 20 pharmacies (Austin: 10, Houston: 10)
- **Success Rate**: 100% (all actor calls succeeded)
- **Credit Usage**: Controlled and predictable with optimization

### 🔧 Optimizations Implemented
- **Cost Control**: Added `forceExit: true` and `maxCrawledPlacesPerSearch` limits
- **Configuration**: Updated `trial_config.json` with `max_results_per_query: 13`
- **Performance**: Eliminated all 400 error responses
- **Data Quality**: Focused on independent pharmacies, filtered out chains

### 🎯 Trial Run Command
```bash
APIFY_ACTOR_ID=nwua9Gu5YrADL7ZDj python scripts/run_trial.py
```

**Output**: Results saved to `data/trial_results/` with combined and per-city JSON files

---

## TDD Implementation Status

### ✅ Completed with TDD
1. **Apify Collector**
   - Enhanced test coverage with proper mocking
   - Fixed actor.call() signature handling
   - Improved test isolation and reliability
   - Added support for structured location-based queries
   - Fixed configuration format handling

2. **Perplexity API Client**
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

---

## 🧠 Core Modules (`src/` Package)

### 📊 Classification System (`src/classification/`)
**Purpose**: AI-powered and rule-based pharmacy classification to distinguish independent pharmacies from chains

#### `classifier.py` - Main Classification Engine
- **Hybrid Approach**: Combines LLM-based (Perplexity API) and rule-based classification
- **Caching**: Integrated caching system to minimize API costs and improve performance
- **Batch Processing**: `batch_classify_pharmacies()` for efficient bulk classification
- **Fallback Logic**: Rule-based classifier serves as backup when LLM fails
- **Key Functions**:
  - `classify_pharmacy()` - Single pharmacy classification with caching
  - `rule_based_classify()` - Fast rule-based classification using chain identifiers
  - `batch_classify_pharmacies()` - Process multiple pharmacies efficiently

#### `perplexity_client.py` - LLM Integration
- **Perplexity API Client**: Professional LLM service for accurate pharmacy classification
- **Rate Limiting**: Built-in request throttling to respect API limits
- **Retry Logic**: Automatic retry with exponential backoff for failed requests
- **Error Handling**: Comprehensive error handling for network/API issues
- **Token Optimization**: Efficient prompt engineering to minimize token usage

#### `cache.py` - Performance Optimization
- **TTL Cache**: Time-based cache expiration to ensure data freshness
- **File Persistence**: Cache survives across application restarts
- **Size Management**: Automatic cache eviction when size limits exceeded
- **Hit/Miss Tracking**: Built-in analytics for cache performance monitoring

### 🔄 Deduplication & Self-Healing (`src/dedup_self_heal/`)
**Purpose**: Intelligent data quality management and automatic gap-filling

#### `dedup.py` - Core Deduplication Logic
- **Smart Deduplication**: Removes duplicate pharmacies while preserving highest confidence entries
- **State Grouping**: Organizes pharmacies by state for targeted processing
- **Gap Detection**: Identifies states with insufficient pharmacy counts
- **Self-Healing**: Automatically triggers additional data collection for under-filled states
- **City-Based Strategy**: Uses major cities for targeted scraping when states need more pharmacies
- **Key Functions**:
  - `process_pharmacies()` - Main orchestration function
  - `remove_duplicates()` - Advanced duplicate detection and removal
  - `identify_underfilled_states()` - Gap analysis for state-level targets
  - `self_heal_state()` - Automatic pharmacy discovery for specific states
  - `merge_new_pharmacies()` - Intelligent merging of new data with existing

#### `apify_integration.py` - Scraper Integration
- **Apify Actor Management**: Handles Google Maps scraper actor calls
- **Query Optimization**: Constructs effective search queries for specific cities
- **Result Processing**: Converts raw Apify results to standardized pharmacy data
- **Error Recovery**: Robust error handling for failed scraping attempts
- **Cost Control**: Integrated with credit tracking to manage budget

### 🛠️ Utilities (`src/utils/`)

#### `api_usage_tracker.py` - Budget Management
- **Credit Tracking**: Real-time monitoring of API credit consumption
- **Budget Enforcement**: Prevents operations that would exceed budget limits
- **Daily Limits**: Optional daily spending caps for cost control
- **Usage Analytics**: Detailed logging and reporting of API usage patterns
- **Global Instance**: `default_tracker` for easy project-wide integration
- **Key Features**:
  - `CreditLimitExceededError` - Exception for budget overruns
  - Daily and total budget tracking
  - Automatic usage logging and persistence

### ⚙️ Configuration (`src/config.py` & `src/__init__.py`)
- **Package Configuration**: Central configuration management
- **API Exports**: Clean interface for accessing classification functions
- **Version Management**: Package versioning and metadata
- **Backward Compatibility**: Maintains compatibility with existing scripts

### 🔗 Integration with Scripts
The `src/` modules seamlessly integrate with the `scripts/` directory:
- **`scripts/apify_collector.py`** uses `src.dedup_self_heal` for data quality
- **Classification workflow** leverages `src.classification` for AI-powered filtering
- **Budget management** through `src.utils.api_usage_tracker` across all scripts
- **Caching system** reduces costs and improves performance project-wide

---

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