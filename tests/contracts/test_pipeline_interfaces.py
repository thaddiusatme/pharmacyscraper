import json
from pathlib import Path

from pharmacy_scraper.orchestrator.pipeline_orchestrator import PipelineOrchestrator


def test_orchestrator_stage_contract_persists_and_returns(tmp_path):
    """Contract: _execute_stage should return stage output and persist it to output_dir.
    Test uses plugin_mode to avoid external deps.
    """
    cfg = {
        "plugin_mode": True,
        "api_keys": {},
        "output_dir": str(tmp_path / "out"),
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))

    orch = PipelineOrchestrator(str(cfg_path))
    orch.state_manager.reset_state()

    def stage():
        return [1, 2, 3]

    result = orch._execute_stage("classification", stage)
    assert result == [1, 2, 3]

    persisted = Path(cfg["output_dir"]) / "stage_classification_output.json"
    assert persisted.exists()
    data = json.loads(persisted.read_text())
    assert data == [1, 2, 3]
