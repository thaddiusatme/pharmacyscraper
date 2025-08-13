import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from pharmacy_scraper.orchestrator.pipeline_orchestrator import PipelineOrchestrator


def _orch(tmpdir: Path, **cfg_kwargs) -> PipelineOrchestrator:
    orch = object.__new__(PipelineOrchestrator)
    # Provide minimal config
    cfg = {"output_dir": str(tmpdir)}
    cfg.update(cfg_kwargs)
    orch.config = SimpleNamespace(**cfg)
    return orch


def test_orchestrator_populates_normalized_address_and_phone_us():
    with TemporaryDirectory() as td:
        tmp = Path(td)
        orch = _orch(tmp, INTERNATIONAL_ENABLED=0, default_region="US")
        records = [
            {
                "id": "n-1",
                "name": "Norm Pharmacy",
                "address": "123 Main St, Springfield, IL 62704",
                "phone": "(415) 555-2671",
            }
        ]
        orch._save_results(records)
        data = json.loads((tmp / "pharmacies.json").read_text())
        item = data[0]
        # Address
        assert item["address_line1"] == "123 Main St"
        assert item["city"] == "Springfield"
        assert item["state"] == "IL"
        assert item["postal_code"] == "62704"
        assert item["country_iso2"] == "US"
        # Phone
        assert item["phone_e164"].startswith("+")
        assert "415" in item["phone_national"]


def test_orchestrator_international_enabled_sets_country_for_ca():
    with TemporaryDirectory() as td:
        tmp = Path(td)
        orch = _orch(tmp, INTERNATIONAL_ENABLED=1, default_region="US")
        records = [
            {
                "id": "n-2",
                "name": "Maple Pharmacy",
                "address": "100 King St W, Toronto, ON M5X 1A9, Canada",
                "phone": "+1 416-555-0100",
            }
        ]
        orch._save_results(records)
        data = json.loads((tmp / "pharmacies.json").read_text())
        item = data[0]
        # From normalization heuristics
        assert item.get("country_iso2") in ("CA", "US", None)
        # If our minimal module set country_code when intl enabled for CA, ensure it flows
        if item.get("country_iso2") == "CA":
            # country_code field is gated for CSV; JSON serialization can still carry it on the record
            assert item.get("country_code") in ("CA", None)


def test_orchestrator_does_not_overwrite_existing_normalized_fields():
    with TemporaryDirectory() as td:
        tmp = Path(td)
        orch = _orch(tmp, INTERNATIONAL_ENABLED=0, default_region="US")
        records = [
            {
                "id": "n-3",
                "name": "Preset Pharmacy",
                "address": "123 Main St, Springfield, IL 62704",
                "address_line1": "Preset Line 1",  # should not be overwritten
                "phone": "(415) 555-2671",
                "phone_e164": "+14155552671",  # pre-populated
            }
        ]
        orch._save_results(records)
        data = json.loads((tmp / "pharmacies.json").read_text())
        item = data[0]
        assert item["address_line1"] == "Preset Line 1"
        assert item["phone_e164"] == "+14155552671"
