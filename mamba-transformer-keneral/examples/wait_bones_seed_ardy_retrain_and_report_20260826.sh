#!/usr/bin/env bash
set -euo pipefail

ROOT=/data3/zts/MotionMamba2.0/mamba-transformer-keneral
PY=/data3/zts/conda_envs/ardy_env/bin/python
BASE=/data4/zts/result
TRAIN=$BASE/bones_seed_ardy_retrain_lr1e5_20260826_original
LOG=$BASE/bones_seed_ardy_retrained_pipeline_20260826.log

while [[ ! -f "$TRAIN/RETRAIN_COMPLETE" ]]; do
  if ! tmux has-session -t bones_ardy_retrain_20260826 2>/dev/null; then
    echo "retraining session ended before completion" >&2
    exit 1
  fi
  sleep 300
done

cd "$ROOT"
bash examples/run_bones_seed_ardy_retrained_eval_20260826.sh
"$PY" examples/write_bones_seed_contract_report.py
"$PY" examples/write_humanml3d_official_table5_report.py

PYTHONPATH=/data3/zts/packages/mamba_op_triton:/data3/zts/packages/state-spaces-mamba:$ROOT/src \
  "$PY" -m pytest -q \
  tests/test_cache_history_contract.py \
  tests/test_ardy_dit_sequence.py \
  tests/test_ardy_dit_mamba.py

"$PY" - <<'PY'
import json
from pathlib import Path

base = Path("/data4/zts/result")
metrics = json.loads(
    (base / "bones_seed_ardy_retrain_lr1e5_20260826_original/metrics.json").read_text()
)
assert metrics["status"] == "PASS"
assert metrics["steps"] == 50_000
assert metrics["final_eval_loss"] <= 4.0
assert metrics["root_output_weight_norm"] >= 2.0
for path in (
    Path("docs/BONES_SEED_CONTRACTFIX_DUAL_EVAL_20260824.md"),
    Path("docs/HUMANML3D_CONTRACTFIX_DUAL_EVAL_20260823.md"),
):
    assert path.is_file() and path.stat().st_size > 0
print("final_pipeline_validation=PASS")
PY

date -Is > "$BASE/bones_seed_ardy_retrained_long_table5_20260826/REPORT_COMPLETE"
echo "pipeline_complete $(date -Is)" | tee -a "$LOG"
