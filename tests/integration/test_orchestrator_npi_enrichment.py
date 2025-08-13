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


def test_orchestrator_populates_npi_contact_fields():
    """Test that orchestrator populates contact fields from NPI data."""
    with TemporaryDirectory() as td:
        tmp = Path(td)
        orch = _orch(tmp)
        records = [
            {
                "id": "npi-1",
                "name": "Test Pharmacy",
                "npi_data": {
                    "basic": {
                        "authorized_official_first_name": "Jane",
                        "authorized_official_last_name": "Smith",
                        "authorized_official_title_or_credential": "PharmD"
                    }
                }
            }
        ]
        orch._save_results(records)
        data = json.loads((tmp / "pharmacies.json").read_text())
        item = data[0]
        
        assert item["contact_name"] == "Jane Smith"
        assert item["contact_role"] == "PharmD"  
        assert item["contact_source"] == "npi_authorized_official"


def test_orchestrator_preserves_existing_contact_fields():
    """Test that orchestrator does not overwrite existing contact fields."""
    with TemporaryDirectory() as td:
        tmp = Path(td)
        orch = _orch(tmp)
        records = [
            {
                "id": "npi-2", 
                "name": "Test Pharmacy",
                "contact_name": "Pre-existing Contact",
                "contact_role": "Manager",
                "contact_source": "api",
                "npi_data": {
                    "basic": {
                        "authorized_official_first_name": "Jane",
                        "authorized_official_last_name": "Smith",
                        "authorized_official_title_or_credential": "PharmD"
                    }
                }
            }
        ]
        orch._save_results(records)
        data = json.loads((tmp / "pharmacies.json").read_text())
        item = data[0]
        
        # Should preserve existing values
        assert item["contact_name"] == "Pre-existing Contact"
        assert item["contact_role"] == "Manager"
        assert item["contact_source"] == "api"


def test_orchestrator_handles_missing_npi_data():
    """Test that orchestrator gracefully handles records without NPI data."""
    with TemporaryDirectory() as td:
        tmp = Path(td)
        orch = _orch(tmp)
        records = [
            {
                "id": "no-npi",
                "name": "Test Pharmacy",
                # no npi_data field
            }
        ]
        orch._save_results(records)
        data = json.loads((tmp / "pharmacies.json").read_text())
        item = data[0]
        
        # Should have None values (set by schema projection)
        assert item["contact_name"] is None
        assert item["contact_role"] is None
        assert item["contact_source"] is None


def test_orchestrator_fills_partial_contact_from_npi():
    """Test that NPI data fills in missing contact fields."""
    with TemporaryDirectory() as td:
        tmp = Path(td)
        orch = _orch(tmp)
        records = [
            {
                "id": "partial-contact",
                "name": "Test Pharmacy", 
                "contact_name": "Existing Name",
                # missing contact_role and contact_source
                "npi_data": {
                    "basic": {
                        "authorized_official_first_name": "Jane",
                        "authorized_official_last_name": "Smith", 
                        "authorized_official_title_or_credential": "PharmD"
                    }
                }
            }
        ]
        orch._save_results(records)
        data = json.loads((tmp / "pharmacies.json").read_text())
        item = data[0]
        
        # Should preserve existing name but fill role
        assert item["contact_name"] == "Existing Name"
        assert item["contact_role"] == "PharmD"
        # contact_source stays None since existing name wasn't from NPI
        assert item["contact_source"] is None
