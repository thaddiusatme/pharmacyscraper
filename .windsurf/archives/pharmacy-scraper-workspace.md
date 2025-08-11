---
trigger: always_on
---

# Pharmacy Scraper Cascade Rules Configuration
# Version: 1.0.0
# Last Updated: 2025-06-26

# Rule Set for Pharmacy Scraper Project
rules:
  # Template Assistance Rules
  - name: template-assistance
    description: Provide template-based assistance for common tasks
    triggers:
      - intent: create_issue
        patterns: 
          - "create (bug|issue|ticket)"
          - "report a problem"
          - "found an? (issue|bug)"
        priority: high
      - intent: new_feature
        patterns:
          - "add feature"
          - "implement new"
          - "create new (feature|functionality)"
        priority: high
      - intent: update_docs
        patterns:
          - "update documentation"
          - "document this"
          - "add docs"
        priority: medium
    actions:
      - name: suggest_template
        template: "${intent}_template.md"
        description: "Suggested template for ${intent}"
      - name: provide_guidance
        message: |
          Consider including:
          - Clear description of the ${intent}
          - Steps to reproduce (for bugs)
          - Expected vs actual behavior
          - Relevant code snippets
          - Test cases
          - Documentation updates

  # Code Quality Rules
  - name: code-quality-checks
    description: Enforce code quality standards
    triggers:
      - file_change: "*.py"
        actions: [create, modify]
    actions:
      - name: run_linter
        command: "flake8 ${file}"
        on_failure: "warn"
      - name: check_type_hints
        enabled: true
        message: "Consider adding type hints to function signatures"
      - name: suggest_docstrings
        enabled: true
        patterns: 
          - "def "
        message: "Remember to add/update docstrings for functions and classes"

  # Documentation Rules
  - name: documentation-updater
    description: Keep documentation in sync with code changes
    triggers:
      - code_change: "*.py"
        patterns:
          - "def "
          - "class "
    actions:
      - name: check_docs
        message: "Check if documentation needs updating for this change"
      - name: suggest_updates
        template: "documentation_update.md"
        when: "major_change"

  # Test Coverage Rules
  - name: test-coverage
    description: Ensure adequate test coverage
    triggers:
      - file_change: "src/**/*.py"
        exclude: ["**/__init__.py", "**/tests/**"]
    actions:
      - name: check_test_file
        pattern: "${file/%.py/_test.py}"
        message: "Consider adding/updating tests for this file"
      - name: suggest_test_cases
        when: "new_function"
        message: "Add test cases for the new function"

  # Project-Specific Rules
  - name: pharmacy-scraper-specific
    description: Rules specific to Pharmacy Scraper project
    rules:
      - name: api-key-safety
        description: Prevent accidental API key exposure
        triggers:
          - pattern: "(?:api[_-]?key|token|secret)"
            in: [file_content, commit_message]
        actions:
          - name: warn
            message: "Potential API key detected. Never commit sensitive information!"
            level: "error"

      - name: classification-accuracy
        description: Monitor classification accuracy
        triggers:
          - pattern: "classify_pharmacy"
            in: [code_change]
        actions:
          - name: suggest_test_cases
            message: "Add test cases for new classification rules"
          - name: check_metrics
            message: "Verify classification metrics after changes"

# Rule Priorities
priorities:
  - name: high
    color: "#FF6B6B"
    description: "Critical issues that need immediate attention"
  - name: medium
    color: "#FFD166"
    description: "Important but not critical"
  - name: low
    color: "#06D6A0"
    description: "Nice to have improvements"

# Template Locations
templates:
  feature_request: ".github/ISSUE_TEMPLATE/feature_request.md"
  bug_report: ".github/ISSUE_TEMPLATE/bug_report.md"
  documentation_update: ".github/ISSUE_TEMPLATE/documentation_update.md"

# Ignore Patterns
ignore:
  - "**/__pycache__/**"
  - "**/.pytest_cache/**"
  - "**/venv/**"
  - "**/node_modules/**"
  - "**/.git/**"
  - "**/*.min.js"
  - "**/*.min.css"

# Version Control
version_control:
  branch_protection:
    main:
      required_reviews: 1
      require_passing_tests: true
      require_linear_history: true
    develop:
      required_reviews: 1
      require_passing_tests: true

# Auto-Formatting
formatting:
  python:
    formatter: "black"
    line_length: 88
  markdown:
    formatter: "prettier"
    options: "--prose-wrap always"

# Linting
linting:
  python:
    linter: "flake8"
    config: ".flake8"
  markdown:
    linter: "markdownlint"
    config: ".markdownlint.json"

# Testing
testing:
  framework: "pytest"
  pattern: "**/*_test.py"
  coverage:
    minimum: 80
    report: "term-missing"

# Documentation
documentation:
  generate:
    enabled: true
    output: "docs/api"
    style: "google"  # Google-style docstrings
  live_preview: true

# Notifications
notifications:
  on_failure: "error"  # or 'warn', 'info', 'none'
  on_success: "info"
  channels:
    - "ide"
    - "cli"
    - "github"

# Memory Management
memory:
  context_window: 8000  # tokens
  max_items: 10
  priority: ["recent", "frequent"]

# Tool Configuration
tools:
  code_search:
    max_results: 5
    include: ["src/**/*.py", "tests/**/*.py"]
    exclude: ["**/__pycache__/**"]
  file_operations:
    confirm_overwrite: true
    create_backup: true

# Environment Variables
environment:
  required:
    - "PYTHONPATH"
  optional:
    - "APIFY_TOKEN"
    - "GOOGLE_PLACES_API_KEY"
    - "PERPLEXITY_API_KEY"

# Security
security:
  secrets_detection: true
  banned_patterns:
    - "(?:password|secret|api[_-]?key|token)\\s*[=:]",
  allowed_domains:
    - "api.apify.com"
    - "maps.googleapis.com"
    - "api.perplexity.ai"
