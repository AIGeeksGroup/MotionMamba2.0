#!/usr/bin/env bash
set -euo pipefail

ROOT=/data3/zts/MotionMamba2.0/mamba-transformer-keneral
PY=/data3/zts/conda_envs/ardy_env/bin/python
BASE=/data4/zts/result
CACHE=$BASE/bones_seed_text_cache_llm2vec
STATE=$BASE/bones_seed_end_to_end_20260824
mkdir -p "$STATE"

export HF_HOME=/data3/zts/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LOCAL_CACHE=true
export PYTHONPATH=/data3/zts/packages/mamba_op_triton:/data3/zts/packages/state-spaces-mamba:/data3/zts/MotionMamba2.0/mamba-transformer-keneral/src${PYTHONPATH:+:$PYTHONPATH}

CURRENT_STAGE=initializing
on_error() {
  local rc=$?
  echo "$(date -Is) stage=$CURRENT_STAGE status=FAIL exit_code=$rc" \
    | tee -a "$STATE/pipeline.log" "$STATE/FAILED"
  exit "$rc"
}
trap on_error ERR

log_stage() {
  CURRENT_STAGE=$1
  echo "$(date -Is) stage=$1" | tee -a "$STATE/pipeline.log"
}

validate_training() {
  local tag=$1 expected_steps=$2
  "$PY" - "$BASE" "$tag" "$expected_steps" <<'PY'
import json
import sys
from pathlib import Path

base, tag, expected_steps = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
for mode in ("original", "minimal", "hybrid"):
    if tag == "contract_full50k_20260824":
        run = base / f"bones_seed_{tag}_{mode}"
    else:
        run = base / f"bones_seed_{tag}_{mode}"
    metrics_path = run / "metrics.json"
    checkpoint = run / f"{mode}_humanml3d_ddp.pt"
    metrics = json.loads(metrics_path.read_text())
    assert metrics["status"] == "PASS", (mode, metrics["status"])
    assert metrics["steps"] == expected_steps, (mode, metrics["steps"])
    assert metrics["all_gradients_finite"] is True, mode
    assert checkpoint.is_file() and checkpoint.stat().st_size > 0, checkpoint
print(f"training_validation=PASS tag={tag} steps={expected_steps}")
PY
}

log_stage waiting_for_data_build
while [[ ! -f "$CACHE/complete.json" ]]; do
  sleep 60
done
"$PY" - "$BASE" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

base = Path(sys.argv[1])
cache = base / "bones_seed_text_cache_llm2vec"
complete = json.loads((cache / "complete.json").read_text())
assert complete["status"] == "PASS", complete
expected = 0
manifest_ids = []
for split in ("train", "val", "test"):
    summary = json.loads((base / f"bones_seed_ardy_{split}" / "summary.json").read_text())
    assert summary["status"] == "PASS", (split, summary)
    assert summary["skipped"]["invalid"] == 0, (split, summary["skipped"])
    expected += summary["samples"]
    with (base / f"bones_seed_ardy_{split}" / "manifest.jsonl").open() as handle:
        manifest_ids.extend(json.loads(line)["id"] for line in handle)
assert complete["samples"] == expected, (complete["samples"], expected)
progress = json.loads((cache / "progress.json").read_text())
assert progress == {"completed": expected, "total": expected}, progress
assert len(manifest_ids) == len(set(manifest_ids)) == expected
index = json.loads((cache / "index.json").read_text())
samples = index["samples"]
assert len(samples) == expected
assert [sample["row"] for sample in samples] == list(range(expected))
assert [sample["id"] for sample in samples] == manifest_ids
embeddings = np.load(cache / "embeddings.npy", mmap_mode="r")
assert embeddings.shape == (expected, 4096), embeddings.shape
for start in range(0, expected, 4096):
    block = embeddings[start:min(start + 4096, expected)]
    assert np.isfinite(block).all(), f"non-finite embedding block at row {start}"
    assert np.any(block != 0, axis=1).all(), f"zero embedding row in block at {start}"
print(f"data_validation=PASS samples={expected}")
PY

if validate_training gate1k 1000 >/dev/null 2>&1; then
  log_stage gate_1k_already_pass
else
  log_stage gate_1k_start
  STEPS=1000 TAG=gate1k "$ROOT/examples/run_bones_seed_contract_gate_20260824.sh"
  validate_training gate1k 1000
  log_stage gate_1k_pass
fi

if validate_training gate5k 5000 >/dev/null 2>&1; then
  log_stage gate_5k_already_pass
else
  log_stage gate_5k_start
  STEPS=5000 TAG=gate5k PARENT_TAG=gate1k \
    "$ROOT/examples/run_bones_seed_contract_gate_20260824.sh"
  validate_training gate5k 5000
  log_stage gate_5k_pass
fi

if validate_training contract_full50k_20260824 50000 >/dev/null 2>&1; then
  log_stage full_50k_already_pass
else
  log_stage full_50k_start
  "$ROOT/examples/run_bones_seed_contract_full50k_20260824.sh"
  validate_training contract_full50k_20260824 50000
  log_stage full_50k_pass
fi

log_stage table45_eval_start
"$ROOT/examples/run_bones_seed_contract_dual_eval_20260824.sh"
"$PY" - "$BASE/bones_seed_contract_dual_eval_20260824" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
by_protocol = {}
for protocol in ("table4", "table5_in", "table5_out"):
    by_protocol[protocol] = []
    for mode in ("original", "minimal", "hybrid"):
        path = root / f"{mode}_{protocol}.json"
        result = json.loads(path.read_text())
        assert result["status"] == "PASS", (mode, protocol, result["status"])
        assert result["samples"] == 64, (mode, protocol, result["samples"])
        by_protocol[protocol].append(result["sample_ids"])
    assert by_protocol[protocol][0] == by_protocol[protocol][1] == by_protocol[protocol][2], protocol
assert by_protocol["table5_in"][0] == by_protocol["table5_out"][0], "Table 5 in/out panels differ"
print("table45_validation=PASS jobs=9 sample_ids=matched")
PY
log_stage table45_eval_pass
log_stage report_generation_start
"$PY" "$ROOT/examples/write_bones_seed_contract_report.py"
test -s "$ROOT/docs/BONES_SEED_CONTRACTFIX_DUAL_EVAL_20260824.md"
log_stage report_generation_pass
touch "$STATE/READY_FOR_REPORT"
log_stage ready_for_report
