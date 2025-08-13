import json
from pathlib import Path

from pharmacy_scraper.pipeline.plugin_pipeline import run_pipeline


def test_vet_clinic_classifier_with_csv_source(tmp_path: Path):
    # Prepare a simple CSV with one vet clinic-like row
    csv_path = tmp_path / "vet.csv"
    csv_path.write_text("name,address\nHappy Tails Vet Clinic,123 Pet St\n")

    config = {
        "plugins": {
            "sources": [
                "pharmacy_scraper.plugins.csv_source:CSVSourcePlugin",
            ],
            "classifiers": [
                "pharmacy_scraper.plugins.example_vet_clinic:VetClinicClassifierPlugin",
            ],
        },
        "plugin_config": {
            "CSVSourcePlugin": {
                "path": str(csv_path),
            }
        },
    }

    # Include a business_type in the query for future extensibility
    out = run_pipeline(config, query={"business_type": "vet_clinic"})
    assert isinstance(out, list) and len(out) == 1
    row = out[0]
    # Ensure classifier label and source fields preserved
    assert row.get("label") == "vet_clinic"
    assert row.get("name") == "Happy Tails Vet Clinic"
