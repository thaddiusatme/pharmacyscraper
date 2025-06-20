# Independent Pharmacy Verification Project

## Project Overview
This project collects and verifies information about independent pharmacies across the United States.

## Setup

1. Clone the repository
   ```bash
   git clone <repository-url>
   cd pharmacy-verification
   ```

2. Set up the environment
   ```bash
   # Run the setup script
   ./setup.sh
   
   # Activate virtual environment
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. Configure environment variables
   - Copy `.env.example` to `.env`
   - Add your API keys and configuration

## Project Structure

```
.
├── data/               # Data files
│   ├── raw/            # Raw data from sources
│   └── processed/      # Processed and cleaned data
├── scripts/            # Python scripts
├── logs/               # Log files
├── reports/            # Generated reports
├── docs/               # Documentation
├── .gitignore
├── requirements.txt
└── README.md
```

## Usage

1. Run data collection:
   ```bash
   python scripts/collect_pharmacies.py
   ```

2. Run verification:
   ```bash
   python scripts/verify_pharmacies.py
   ```

3. Generate reports:
   ```bash
   python scripts/generate_report.py
   ```

## License
MIT
