import csv
from pathlib import Path


def test_csv_source_plugin_reads_rows(tmp_path):
    # Create a sample CSV
    csv_path = tmp_path / "pharmacies.csv"
    rows = [
        {"name": "Indie Pharmacy", "address": "123 Main St"},
        {"name": "CVS Pharmacy", "address": "456 Elm Ave"},
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "address"])
        writer.writeheader()
        writer.writerows(rows)

    from pharmacy_scraper.plugins.registry import load_class
    Adapter = load_class("pharmacy_scraper.plugins.csv_source:CSVSourcePlugin")

    plugin = Adapter()
    out = plugin.fetch(query={}, cfg={"path": str(csv_path)})

    assert isinstance(out, list) and len(out) == 2
    assert out[0]["name"] == "Indie Pharmacy"


def test_csv_source_plugin_raises_without_path(tmp_path):
    from pharmacy_scraper.plugins.registry import load_class
    Adapter = load_class("pharmacy_scraper.plugins.csv_source:CSVSourcePlugin")
    plugin = Adapter()
    try:
        plugin.fetch(query={}, cfg={})
        assert False, "expected ValueError when 'path' missing"
    except ValueError:
        pass
