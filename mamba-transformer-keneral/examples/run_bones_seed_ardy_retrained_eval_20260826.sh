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
TRAIN_OUT=$BASE/bones_seed_ardy_retrain_lr1e5_20260826_original
TABLE4_OUT=$BASE/bones_seed_ardy_retrained_dual_eval_20260826
TABLE5_OUT=$BASE/bones_seed_ardy_retrained_official_table5_20260826
LONG_OUT=$BASE/bones_seed_ardy_retrained_long_table5_20260826
MANIFEST=$BASE/bones_seed_ardy_test/manifest.jsonl
STATS=$BASE/bones_seed_ardy_train
CACHE=$BASE/bones_seed_text_cache_llm2vec

ARDY=$TRAIN_OUT/original_humanml3d_ddp.pt
MINIMAL=$BASE/bones_seed_contract_full50k_20260824_minimal/minimal_humanml3d_ddp.pt
HYBRID=$BASE/bones_seed_contract_full50k_20260824_hybrid/hybrid_humanml3d_ddp.pt

mkdir -p "$TABLE4_OUT" "$TABLE5_OUT" "$LONG_OUT"
[[ -f "$TRAIN_OUT/RETRAIN_COMPLETE" ]] || { echo "ARDY retraining is incomplete" >&2; exit 1; }

checkpoint_for() {
  case "$1" in
    original) echo "$ARDY" ;;
    minimal) echo "$MINIMAL" ;;
    hybrid) echo "$HYBRID" ;;
    *) return 1 ;;
  esac
}

result_pass() {
  local path=$1 expected_samples=$2
  [[ -f "$path" ]] || return 1
  "$PY" - "$path" "$expected_samples" <<'PY'
import json
import sys
from pathlib import Path

path, expected = Path(sys.argv[1]), int(sys.argv[2])
result = json.loads(path.read_text())
raise SystemExit(0 if result.get("status") == "PASS" and result.get("samples") == expected else 1)
PY
}

run_table4() {
  local gpu=$1 mode=$2 checkpoint stem
  checkpoint=$(checkpoint_for "$mode")
  stem=${mode}_table4
  if result_pass "$TABLE4_OUT/$stem.json" 64 && [[ -f "$TABLE4_OUT/$stem.npz" ]]; then
    echo "skip_completed $stem"
    return
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$ROOT/examples/evaluate_humanml3d_table45.py" \
    --dataset-name BONES-SEED-v004 --model g1 --mode "$mode" \
    --checkpoint "$checkpoint" --manifest "$MANIFEST" --stats-dir "$STATS" \
    --text-cache "$CACHE" --protocol table4 --samples 64 --batch-size 4 \
    --diffusion-steps 10 --cfg-weight-text 2 --cfg-weight-constraint 2 \
    --seed 20260824 --device cuda:0 --output "$TABLE4_OUT/$stem.json" \
    --artifacts-output "$TABLE4_OUT/$stem.npz" >"$TABLE4_OUT/$stem.log" 2>&1
}

run_table5() {
  local gpu=$1 mode=$2 protocol=$3 seconds=$4 samples=$5 selection_seconds=$6 output=$7
  local checkpoint stem
  checkpoint=$(checkpoint_for "$mode")
  stem=${mode}_${seconds}s_${protocol}
  if result_pass "$output/$stem.json" "$samples" && [[ -f "$output/$stem.npz" ]]; then
    echo "skip_completed $stem"
    return
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$ROOT/examples/evaluate_ardy_official_table5.py" \
    --dataset-name BONES-SEED-v004 --model g1 --mode "$mode" \
    --checkpoint "$checkpoint" --manifest "$MANIFEST" --stats-dir "$STATS" \
    --text-cache "$CACHE" --protocol "$protocol" --total-seconds "$seconds" \
    --selection-min-seconds "$selection_seconds" --samples "$samples" --batch-size 2 \
    --diffusion-steps 10 --cfg-weight-text 2 --cfg-weight-constraint 2 \
    --seed 20260824 --device cuda:0 --output "$output/$stem.json" \
    --artifacts-output "$output/$stem.npz" >"$output/$stem.log" 2>&1
}

wait_group() {
  local failed=0 pid
  for pid in "$@"; do
    wait "$pid" || failed=1
  done
  (( failed == 0 ))
}

pids=()
for spec in "0 original" "1 minimal" "2 hybrid"; do
  read -r gpu mode <<<"$spec"
  run_table4 "$gpu" "$mode" & pids+=("$!")
done
wait_group "${pids[@]}"

pids=()
index=0
gpus=(0 1 2 3 4 6)
for protocol in in_horizon out_of_horizon; do
  for mode in original minimal hybrid; do
    run_table5 "${gpus[$index]}" "$mode" "$protocol" 9 64 9 "$TABLE5_OUT" &
    pids+=("$!")
    index=$((index + 1))
  done
done
wait_group "${pids[@]}"

for seconds in 20 40; do
  pids=()
  index=0
  for protocol in in_horizon out_of_horizon; do
    for mode in original minimal hybrid; do
      run_table5 "${gpus[$index]}" "$mode" "$protocol" "$seconds" 20 40 "$LONG_OUT" &
      pids+=("$!")
      index=$((index + 1))
    done
  done
  wait_group "${pids[@]}"
done

"$PY" - "$TABLE4_OUT" "$TABLE5_OUT" "$LONG_OUT" <<'PY'
import json
import sys
from pathlib import Path

table4, table5, long_root = map(Path, sys.argv[1:])
modes = ("original", "minimal", "hybrid")
protocols = ("in_horizon", "out_of_horizon")
for mode in modes:
    result = json.loads((table4 / f"{mode}_table4.json").read_text())
    assert result["status"] == "PASS" and result["samples"] == 64
for protocol in protocols:
    panels = []
    for mode in modes:
        result = json.loads((table5 / f"{mode}_9s_{protocol}.json").read_text())
        assert result["status"] == "PASS" and result["samples"] == 64
        assert result["generation_windows"] == 4
        panels.append(result["sample_ids"])
    assert panels[0] == panels[1] == panels[2]
long_panels = []
for seconds in (20, 40):
    for protocol in protocols:
        panels = []
        for mode in modes:
            result = json.loads((long_root / f"{mode}_{seconds}s_{protocol}.json").read_text())
            assert result["status"] == "PASS" and result["samples"] == 20
            assert result["selection_min_seconds"] == 40.0
            assert result["protocol_origin"] == "duration_extension"
            panels.append(result["sample_ids"])
        assert panels[0] == panels[1] == panels[2]
        long_panels.append(panels[0])
assert all(panel == long_panels[0] for panel in long_panels[1:])
print("corrected_evaluation_validation=PASS")
PY

date -Is > "$LONG_OUT/EVALUATION_COMPLETE"
echo "evaluation_complete table4=$TABLE4_OUT table5=$TABLE5_OUT long=$LONG_OUT"
