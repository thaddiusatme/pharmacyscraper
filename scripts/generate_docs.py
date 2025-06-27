#!/usr/bin/env python3
"""
Documentation Generator for Pharmacy Scraper

This script generates HTML documentation for the Pharmacy Scraper project.
It uses a template file and injects dynamic content like build status and timestamps.
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

def load_template(template_path: str) -> str:
    """Load the HTML template from file."""
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: Template file not found at {template_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading template: {e}", file=sys.stderr)
        sys.exit(1)

def get_commit_sha() -> str:
    """Get the current git commit SHA if available."""
    try:
        import subprocess
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
    except Exception:
        return "unknown"

def generate_documentation(
    output_dir: str,
    template_path: str,
    workflow_status: str = "unknown",
    test_status: str = "unknown",
    lint_status: str = "unknown",
    security_status: str = "unknown",
    docs_status: str = "unknown"
) -> bool:
    """Generate documentation using the template and provided statuses."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'index.html')
    
    # Get current timestamp
    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
    
    # Get git commit SHA
    commit_sha = get_commit_sha()
    
    # Load template
    template = load_template(template_path)
    
    # Render template with variables
    html_content = template
    
    # Simple template variable substitution
    html_content = html_content.replace('{{ timestamp }}', timestamp)
    html_content = html_content.replace('{{ workflow_status }}', workflow_status)
    html_content = html_content.replace('{{ test_status }}', test_status)
    html_content = html_content.replace('{{ lint_status }}', lint_status)
    html_content = html_content.replace('{{ security_status }}', security_status)
    html_content = html_content.replace('{{ docs_status }}', docs_status)
    html_content = html_content.replace('{{ commit_sha }}', commit_sha)
    
    # Write the output file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"Documentation generated successfully at {output_file}")
        return True
    except Exception as e:
        print(f"Error writing documentation: {e}", file=sys.stderr)
        return False

def main():
    """Main entry point for the documentation generator."""
    parser = argparse.ArgumentParser(description='Generate project documentation.')
    parser.add_argument('--output-dir', default='docs/_build/html',
                       help='Output directory for generated documentation')
    parser.add_argument('--template', default='.github/workflows/templates/documentation.html',
                       help='Path to the documentation template')
    parser.add_argument('--workflow-status', default='unknown',
                       choices=['success', 'failure', 'cancelled', 'unknown'],
                       help='Overall workflow status')
    parser.add_argument('--test-status', default='unknown',
                       choices=['success', 'failure', 'cancelled', 'skipped', 'unknown'],
                       help='Test job status')
    parser.add_argument('--lint-status', default='unknown',
                       choices=['success', 'failure', 'cancelled', 'skipped', 'unknown'],
                       help='Lint job status')
    parser.add_argument('--security-status', default='unknown',
                       choices=['success', 'failure', 'cancelled', 'skipped', 'unknown'],
                       help='Security scan status')
    parser.add_argument('--docs-status', default='success',
                       choices=['success', 'failure', 'cancelled', 'skipped', 'unknown'],
                       help='Documentation generation status')
    
    args = parser.parse_args()
    
    # Generate documentation
    success = generate_documentation(
        output_dir=args.output_dir,
        template_path=args.template,
        workflow_status=args.workflow_status,
        test_status=args.test_status,
        lint_status=args.lint_status,
        security_status=args.security_status,
        docs_status=args.docs_status
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
