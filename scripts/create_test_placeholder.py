#!/usr/bin/env python3
"""
Create a placeholder test results file for CI/CD pipeline.
This script generates a minimal JUnit XML file when tests fail to run.
"""

import os
import sys
from datetime import datetime


def create_placeholder_results(output_path: str, reason: str = "Test execution failed"):
    """Create a placeholder JUnit XML file."""
    xml_content = f"""<?xml version="1.0" encoding="utf-8"?>
<testsuites>
  <testsuite name="pytest" tests="0" errors="1" failures="0" skipped="0" time="0.0" timestamp="{datetime.now().isoformat()}">
    <testcase classname="pipeline" name="test_execution" time="0.0">
      <error message="{reason}">Unable to execute tests or collect results</error>
    </testcase>
  </testsuite>
</testsuites>"""
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write the XML content
    with open(output_path, 'w') as f:
        f.write(xml_content)
    
    print(f"✅ Created placeholder test results: {output_path}")
    return True


def main():
    """Main function to create placeholder test results."""
    if len(sys.argv) < 2:
        print("Usage: python create_test_placeholder.py <output_path> [reason]")
        sys.exit(1)
    
    output_path = sys.argv[1]
    reason = sys.argv[2] if len(sys.argv) > 2 else "Test execution failed"
    
    try:
        create_placeholder_results(output_path, reason)
        print(f"📊 Placeholder test results created successfully")
    except Exception as e:
        print(f"❌ Failed to create placeholder: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
