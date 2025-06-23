# Pharmacy Scraper: Project Digest & Architecture

This document provides a technical overview of the Independent Pharmacy Verification project, detailing its architecture, key components, and operational strategy.

### Project Goals
**Client**: Pharmacy mailing list verification
**Budget**: $350
**Timeline**: 3-4 days
**Deliverable**: Verified list of 1,250 independent pharmacies (25 per state) with mailing addresses

---

## 1. Core Architecture: A Multi-Phase Pipeline

The project is built around a sequential, multi-phase data processing pipeline orchestrated by `scripts/run_pipeline.py`. This design ensures that data flows through distinct stages of collection, cleaning, and analysis in a controlled and reproducible manner.

**Pipeline Stages:**
1.  **Phase 1: Data Collection**: Raw pharmacy data is collected from Google Maps using the Apify platform. Search queries are dynamically generated based on the active configuration file.
2.  **Phase 1.5: Deduplication**: The raw dataset is processed to remove duplicate entries based on unique identifiers like `placeId` and `title`, ensuring data integrity.
3.  **Phase 2a: Classification**: Each unique pharmacy record is classified as either "chain" or "independent" using a hybrid system that combines rule-based logic and a Perplexity API-powered language model.
4.  **Phase 2b: Verification (Optional)**: Address and operational status are verified via the Google Places API. This is an optional, costly step that can be skipped for initial large-scale runs.

---

## 2. Key Components

### a. Orchestration & Execution (`scripts/`)
-   **`run_pipeline.py`**: The main entry point for the entire pipeline. It parses command-line arguments, manages configuration, and calls the various processing modules in sequence.
-   **`apify_collector.py`**: A dedicated module responsible for interfacing with the Apify API, executing actor runs, and retrieving results.

### b. Core Logic (`src/`)

#### `src/classification/` - AI-Powered Classification
- **Purpose**: Distinguishes independent pharmacies from chains.
- **`classifier.py`**: Main engine using a hybrid LLM (Perplexity) and rule-based approach. Features batch processing and caching.
- **`perplexity_client.py`**: Manages Perplexity API interaction with rate limiting, retry logic, and error handling.
- **`cache.py`**: Provides a persistent, TTL-based file cache to minimize API costs and improve performance.

#### `src/dedup_self_heal/` - Data Quality & Gap-Filling
- **Purpose**: Ensures data quality and automatically fills gaps in under-populated states.
- **`dedup.py`**: Implements smart deduplication, identifies states with insufficient data, and triggers a "self-healing" process to collect more.
- **`apify_integration.py`**: Handles the targeted Apify scraper calls required for the self-healing process.

#### `src/verification/` - Address Verification
- **Purpose**: Provides address and operational status verification.
- **`google_places.py`**: Integrates with the Google Places API to verify pharmacy data.

#### `src/utils/` - Shared Utilities
- **Purpose**: Provides project-wide utilities.
- **`api_usage_tracker.py`**: A critical component for budget management, tracking API credit usage in real-time and preventing budget overruns.

### c. Configuration (`config/`)
-   **`trial_config.json`**: For small-scale tests and debugging.
-   **`five_state_run.json`**: A medium-scale configuration for controlled, cost-effective test runs.
-   **`large_scale_run.json`**: The full 50-state configuration for comprehensive national data collection.

---

## 3. Operational Strategy

To ensure robust and cost-effective data collection, the following operational best practices have been implemented:

-   **Phased Rollout**: New features and large-scale runs are first tested on a small scale (e.g., the 5-state run) to validate performance and estimate costs before full deployment.
-   **Persistent Logging**: All pipeline runs generate a detailed log file (e.g., `data/five_state_results/pipeline.log`), capturing all actions, API calls, and errors for debugging and auditing.
-   **Uninterrupted Execution**: On macOS, the `caffeinate` utility is used to prevent the system from sleeping during long-running jobs, ensuring the pipeline runs to completion without interruption.

---

## 4. Recent Milestones (June 2024)

### System Enhancements
- **End-to-End Testing**: Verified the complete workflow from data collection to verification.
- **Test-Driven Development**: Implemented key modules like the Classification System, Apify Collector, and Caching Layer using a TDD approach, achieving high test coverage (e.g., 83% for the Perplexity client).
- **Address Verification Success**: The Google Places integration has demonstrated a 90% verification success rate in trials with a high average confidence score of 0.924.

### Trial Run Optimization
- **Apify Actor Fixes**: Resolved 400 Bad Request errors by fixing invalid input parameters (`allPlacesNoSearchAction`).
- **Cost Control**: Optimized credit usage by implementing `forceExit` and search limits in Apify actor calls.
- **Successful Trial**: A trial run across 4 cities in California and Texas successfully collected 39 independent pharmacies, validating the cost control measures.
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