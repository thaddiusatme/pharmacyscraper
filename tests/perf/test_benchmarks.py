import json
import os
import time
from pathlib import Path
import random
import string
import pytest

from pharmacy_scraper.classification.classifier import rule_based_classify
from pharmacy_scraper.orchestrator.pipeline_orchestrator import PipelineOrchestrator


RUN_PERF = os.getenv("PERF") == "1"
STRICT = os.getenv("PERF_STRICT") == "1"

pytestmark = pytest.mark.skipif(not RUN_PERF, reason="Set PERF=1 to run perf benchmarks")


def _rand_word(n: int = 10) -> str:
    letters = string.ascii_letters
    return "".join(random.choice(letters) for _ in range(n))


def _synthetic_batch(n: int):
    return [
        {"name": f"{_rand_word()} Pharmacy", "address": f"{random.randint(1,9999)} {_rand_word()} St"}
        for _ in range(n)
    ]


def _write_results(tmp_path: Path, name: str, payload: dict):
    out = tmp_path / f"perf_{name}.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return out


def _load_baseline() -> dict:
    here = Path(__file__).parent
    baseline_path = here / "baselines" / "baseline.json"
    if baseline_path.exists():
        return json.loads(baseline_path.read_text())
    return {}


def _regression_check(metric_name: str, value: float, baseline: dict, tolerance_drop: float = 0.15):
    """
    Warn (or fail in STRICT mode) when a metric regresses beyond tolerance.
    For throughput metrics, we consider a drop; for duration we consider an increase.
    We encode sign in baseline config as {"rule_throughput_ops": {"kind":"higher_better", "value": 20000}}
    """
    m = baseline.get(metric_name)
    if not m:
        return  # no baseline to compare
    kind = m.get("kind", "higher_better")
    base = float(m.get("value"))
    if base <= 0:
        return
    if kind == "higher_better":
        # allow drop within tolerance
        drop = (base - value) / base
        if drop > tolerance_drop:
            msg = f"Perf regression: {metric_name} {value:.2f} < baseline {base:.2f} (drop {drop:.1%})"
            if STRICT:
                pytest.fail(msg)
            else:
                pytest.skip(msg)
    else:
        # lower_better
        inc = (value - base) / base
        if inc > tolerance_drop:
            msg = f"Perf regression: {metric_name} {value:.4f}s > baseline {base:.4f}s (increase {inc:.1%})"
            if STRICT:
                pytest.fail(msg)
            else:
                pytest.skip(msg)


def test_rule_based_throughput(tmp_path, monkeypatch):
    # fixed seed for stability
    random.seed(1337)
    batch = _synthetic_batch(5000)

    t0 = time.perf_counter()
    for item in batch:
        _ = rule_based_classify(item)
    dt = time.perf_counter() - t0

    ops = len(batch) / dt if dt > 0 else float('inf')
    results = {"rule_throughput_ops": ops, "duration_s": dt}
    _write_results(tmp_path, "rule_based", results)

    baseline = _load_baseline()
    _regression_check("rule_throughput_ops", ops, baseline)
    _regression_check("rule_duration_s", dt, baseline | {"rule_duration_s": {"kind": "lower_better", "value": baseline.get("rule_duration_s", {}).get("value", dt)}})


def test_orchestrator_stage_timing_plugin_mode(tmp_path):
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
        # tiny payload
        return [1, 2, 3]

    t0 = time.perf_counter()
    _ = orch._execute_stage("classification", stage)
    dt = time.perf_counter() - t0

    results = {"orch_stage_duration_s": dt}
    _write_results(tmp_path, "orch_stage", results)

    baseline = _load_baseline()
    _regression_check("orch_stage_duration_s", dt, baseline | {"orch_stage_duration_s": {"kind": "lower_better", "value": baseline.get("orch_stage_duration_s", {}).get("value", dt)}})
