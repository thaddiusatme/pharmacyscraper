### Epic: Generic Business Data Scraper

1. **As a** data engineer  
   **I want** to pass any list of search terms (e.g., “veterinary clinic”, “car wash”)  
   **So that** the pipeline collects, deduplicates, and outputs matching businesses.

2. **As a** product manager  
   **I want** configuration files to list search terms and target regions  
   **So that** non-technical users can launch new data-collection runs without code changes.

3. **As a** developer  
   **I want** a plug-in interface to add custom classification logic per domain  
   **So that** we can handle domain-specific nuances (e.g., chain detection rules for gas stations).

4. **As an** operations analyst  
   **I want** budget caps enforceable per search term  
   **So that** expensive or runaway terms don’t exceed our total API budget.

5. **As a** tester  
   **I want** mock fixtures for arbitrary business data  
   **So that** CI remains fast and deterministic without real API calls.