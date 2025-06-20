from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pharmacy-verification",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A tool to verify independent pharmacy information",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pharmacy-verification",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.5.0",
        "requests>=2.28.0",
        "googlemaps>=4.5.0",
        "python-dotenv>=0.21.0",
        "apify-client>=1.0.0",
    ],
    entry_points={
        "console_scripts": [
            "pharmacy-verify=scripts.verify_pharmacies:main",
            "pharmacy-collect=scripts.collect_pharmacies:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md"],
    },
)
