#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/data3/zts/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LOCAL_CACHE=true
export PYTHONPATH=/data3/zts/packages/mamba_op_triton:/data3/zts/packages/state-spaces-mamba:/data3/zts/MotionMamba2.0/mamba-transformer-keneral/src${PYTHONPATH:+:$PYTHONPATH}

ROOT=/data3/zts/MotionMamba2.0/mamba-transformer-keneral
PY=/data3/zts/conda_envs/ardy_env/bin/python
BASE=/data4/zts/result
TRAIN_ROOT=$BASE/bones_seed_protocol_aligned_ft_20260826
STAGE=${STAGE:?set STAGE to 500 or 1000}
OUT=$BASE/bones_seed_protocol_aligned_eval_stage${STAGE}_20260826
MANIFEST=$BASE/bones_seed_ardy_test/manifest.jsonl
STATS=$BASE/bones_seed_ardy_train
CACHE=$BASE/bones_seed_text_cache_llm2vec
mkdir -p "$OUT"

checkpoint_for() {
  local mode=$1
  echo "$TRAIN_ROOT/$mode/checkpoint_step_$(printf '%06d' "$STAGE").pt"
}

result_pass() {
  local path=$1 expected=$2
  [[ -f "$path" ]] || return 1
  "$PY" - "$path" "$expected" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text())
raise SystemExit(0 if result.get("status") == "PASS" and result.get("samples") == int(sys.argv[2]) else 1)
PY
}

run_table4() {
  local gpu=$1 mode=$2 checkpoint stem
  checkpoint=$(checkpoint_for "$mode")
  stem=${mode}_table4
  if result_pass "$OUT/$stem.json" 64 && [[ -f "$OUT/$stem.npz" ]]; then
    echo "skip_completed stage=$STAGE $stem"
    return
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$ROOT/examples/evaluate_humanml3d_table45.py" \
    --dataset-name BONES-SEED-v004 --model g1 --mode "$mode" \
    --checkpoint "$checkpoint" --manifest "$MANIFEST" --stats-dir "$STATS" \
    --text-cache "$CACHE" --protocol table4 --samples 64 --batch-size 4 \
    --diffusion-steps 10 --cfg-weight-text 2 --cfg-weight-constraint 2 \
    --seed 20260824 --device cuda:0 --output "$OUT/$stem.json" \
    --artifacts-output "$OUT/$stem.npz" >"$OUT/$stem.log" 2>&1
}

run_duration() {
  local gpu=$1 mode=$2 seconds=$3 protocol=$4 samples=$5 selection=$6 checkpoint stem
  checkpoint=$(checkpoint_for "$mode")
  stem=${mode}_${seconds}s_${protocol}
  if result_pass "$OUT/$stem.json" "$samples" && [[ -f "$OUT/$stem.npz" ]]; then
    echo "skip_completed stage=$STAGE $stem"
    return
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$ROOT/examples/evaluate_ardy_official_table5.py" \
    --dataset-name BONES-SEED-v004 --model g1 --mode "$mode" \
    --checkpoint "$checkpoint" --manifest "$MANIFEST" --stats-dir "$STATS" \
    --text-cache "$CACHE" --protocol "$protocol" --total-seconds "$seconds" \
    --selection-min-seconds "$selection" --samples "$samples" --batch-size 2 \
    --diffusion-steps 10 --cfg-weight-text 2 --cfg-weight-constraint 2 \
    --seed 20260824 --device cuda:0 --output "$OUT/$stem.json" \
    --artifacts-output "$OUT/$stem.npz" >"$OUT/$stem.log" 2>&1
}

run_mode() {
  local mode=$1
  local pids=()
  run_table4 0 "$mode" & pids+=("$!")
  run_duration 1 "$mode" 9 in_horizon 64 9 & pids+=("$!")
  run_duration 2 "$mode" 9 out_of_horizon 64 9 & pids+=("$!")
  run_duration 3 "$mode" 20 in_horizon 20 40 & pids+=("$!")
  run_duration 4 "$mode" 20 out_of_horizon 20 40 & pids+=("$!")
  run_duration 6 "$mode" 40 in_horizon 20 40 & pids+=("$!")
  run_duration 7 "$mode" 40 out_of_horizon 20 40 & pids+=("$!")
  local failed=0 pid
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  (( failed == 0 ))
}

for mode in original minimal hybrid; do
  run_mode "$mode"
done

"$PY" - "$OUT" "$STAGE" <<'PY'
import json
import sys
from pathlib import Path

root, stage = Path(sys.argv[1]), int(sys.argv[2])
modes = ("original", "minimal", "hybrid")
specs = [("table4", 64)] + [
    (f"{seconds}s_{protocol}", 64 if seconds == 9 else 20)
    for seconds in (9, 20, 40)
    for protocol in ("in_horizon", "out_of_horizon")
]
for stem, samples in specs:
    rows = [json.loads((root / f"{mode}_{stem}.json").read_text()) for mode in modes]
    assert all(row["status"] == "PASS" and row["samples"] == samples for row in rows)
    assert all(row["diffusion_steps"] == 10 for row in rows)
    assert all(row["cfg_weight_text"] == row["cfg_weight_constraint"] == 2.0 for row in rows)
    assert rows[0]["sample_ids"] == rows[1]["sample_ids"] == rows[2]["sample_ids"]
    assert all(f"checkpoint_step_{stage:06d}.pt" in row["checkpoint"] for row in rows)
print(f"protocol_aligned_eval_pass stage={stage} jobs={len(specs) * len(modes)}")
PY

date -Is >"$OUT/EVALUATION_COMPLETE"

