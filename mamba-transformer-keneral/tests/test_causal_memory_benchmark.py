import importlib.util
from pathlib import Path

_path = Path(__file__).resolve().parents[1] / "examples" / "causal_memory_benchmark.py"
_spec = importlib.util.spec_from_file_location("causal_memory_benchmark", _path)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)
run = _module.run


def test_cue_is_evicted_and_interventions_are_separable():
    result = run(seed=7, samples=32)
    assert result["status"] == "PASS"
    assert result["cue_evicted_before_w8"] is True
    rows = result["interventions"]
    assert rows["correct"]["cue_accuracy"] == 1.0
    assert rows["minimal"]["cue_accuracy"] == 0.5
    assert rows["zero"]["cue_accuracy"] == 0.5
    assert rows["fresh"]["cue_accuracy"] == 0.5
    assert rows["permute"]["cue_accuracy"] == 0.0
