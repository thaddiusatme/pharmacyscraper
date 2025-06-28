from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README.md
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements from requirements.txt
def load_requirements():
    with open("requirements.txt", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Find all packages under src/
packages = find_packages(where="src")

setup(
    name="pharmacy-scraper",
    version="0.2.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A tool for scraping and verifying independent pharmacy information",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pharmacy-scraper",
    packages=packages,
    package_dir={"": "src"},  # Tell setuptools that packages are under src/
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=load_requirements(),
    # Removed console_scripts since we're using a script-based workflow
    include_package_data=True,
    package_data={
        "pharmacy_scraper": ["*.txt", "*.md", "*.json"],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/pharmacy-scraper/issues",
        "Source": "https://github.com/yourusername/pharmacy-scraper",
    },
)
