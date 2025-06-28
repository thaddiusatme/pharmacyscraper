#!/usr/bin/env python3
"""
Script to organize Cascade and Windsurf rules.
"""
import os
import shutil
from pathlib import Path

def setup_directories():
    """Ensure all required directories exist."""
    dirs = [
        ".github/workflows",
        ".github/linters",
        ".cascade/rules",
        ".windsurf/rules",
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")

def main():
    print("Setting up directories...")
    setup_directories()
    print("\nSetup complete!")  

if __name__ == "__main__":
    main()
