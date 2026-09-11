#!/usr/bin/env bash
set -euo pipefail

# Registered 2x2 structural ablations. This job intentionally waits until the
# final-model selection evaluation has released all GPUs.
ROOT=/data3/zts/MotionMamba2.0/mamba-transformer-keneral
PY=/data3/zts/conda_envs/ardy_env/bin/python
BASE=/data4/zts/result
GATE=$BASE/stage_d2rl_50k_20260906/final_selection_eval/FINAL_SELECTION_EVALUATION_COMPLETE
PARENT=$BASE/bones_seed_contract_full50k_20260824_hybrid/hybrid_humanml3d_ddp.pt
TRAIN=$BASE/bones_seed_ardy_train/manifest.jsonl
VAL=$BASE/bones_seed_ardy_val/manifest.jsonl
STATS=$BASE/bones_seed_ardy_train
TEXT=$BASE/bones_seed_text_cache_llm2vec
OUT=$BASE/stage_d2rl_structural_ablations_20260906

export HF_HOME=/data3/zts/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LOCAL_CACHE=true
export PYTHONPATH=$ROOT/src:/data3/zts/MotionMamba2.0/ardy:/data3/zts/packages/state-spaces-mamba:/data3/zts/packages/mamba_op_triton${PYTHONPATH:+:$PYTHONPATH}

while [[ ! -f "$GATE" ]]; do
  if ! tmux has-session -t d2rl_post 2>/dev/null; then
    echo "final selection tmux session ended before completion marker" >&2
    exit 1
  fi
  sleep 60
done
mkdir -p "$OUT"

COMMON=(
  --model g1 --mode hybrid --manifest "$TRAIN" --dev-manifest "$VAL" --stats-dir "$STATS"
  --text-cache "$TEXT" --steps 50000 --batch-size 2 --learning-rate 1e-4 --seed 9032026
  --frames 312 --history-probability 0 --constraint-dropout 0.1 --text-dropout 0.1
  --sequential-cache-probability 1 --sequential-rollout-min-windows 6 --sequential-rollout-windows 6
  --sequential-rollout-curriculum d2_post_eviction_v1 --generated-history-probability-start 1
  --generated-history-probability-1k 1 --generated-history-probability-5k 1
  --generated-history-diffusion-steps 10 --generated-history-max-supervision-points 1
  --generated-history-bptt-windows 4 --generated-history-attach-seed-write
  --eval-protocol explicit_history --eval-batches 1 --eval-every 5000 --save-every 5000
  --keep-last-checkpoints 2 --boundary-loss-weight 10 --root-velocity-loss-weight 50
  --constraint-loss-weight 1 --fk-constraint-loss-weight 1 --foot-skate-loss-weight 50
  --train-scope state_memory --attention-window-chunks 4 --state-summary-dim 128
  --use-state-delta --state-conditioning-scale-init 0 --state-aware-output-mode native_gated_add
  --init-checkpoint "$PARENT" --log-every 10
)

run_state_query_only() {
  local out="$OUT/state_query_only"
  [[ -f "$out/TRAINING_COMPLETE" ]] && return 0
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES=4,5 "$PY" -m torch.distributed.run --standalone --nproc-per-node=2 \
    "$ROOT/examples/train_humanml3d_ardy_ddp.py" "${COMMON[@]}" \
    --token-state-interaction state_query --output-dir "$out" \
    >"$out/train.log" 2>&1
  date -Is >"$out/TRAINING_COMPLETE"
}

run_memory_only() {
  local out="$OUT/persistent_memory_only"
  [[ -f "$out/TRAINING_COMPLETE" ]] && return 0
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES=6,7 "$PY" -m torch.distributed.run --standalone --nproc-per-node=2 \
    "$ROOT/examples/train_humanml3d_ardy_ddp.py" "${COMMON[@]}" \
    --token-state-interaction naive_gated_add --content-memory-enabled --content-memory-slots 2 \
    --content-memory-use-state-delta --content-memory-write-gate-context previous_memory \
    --content-memory-decoder-init-scale 0.02 --content-memory-write-gate-init-scale 0.02 \
    --content-memory-write-gate-bias-init 0 --content-memory-read-scale-init 0 \
    --content-memory-conditioning text --content-memory-normalize-delta --content-memory-normalize-slots \
    --content-memory-max-write-rate 0.25 --output-dir "$out" \
    >"$out/train.log" 2>&1
  date -Is >"$out/TRAINING_COMPLETE"
}

run_state_query_only & p1=$!
run_memory_only & p2=$!
status=0
wait "$p1" || status=1
wait "$p2" || status=1
date -Is >"$OUT/ABLATION_TRAINING_COMPLETE" || status=1

if (( status == 0 )); then
  PANEL=$BASE/bones_seed_long_memory_panel_v2_20260831/selection
  LOCK=$ROOT/configs/bones_seed_long_memory_protocol_v2.json
  eval_one() {
    local gpu=$1 name=$2 mode=$3 seconds=$4
    local ckpt="$OUT/$name/hybrid_humanml3d_ddp.pt"
    local dir="$OUT/eval/$name/${seconds}s"
    mkdir -p "$dir"
    if [[ "$seconds" == table4 ]]; then
      CUDA_VISIBLE_DEVICES=$gpu "$PY" "$ROOT/examples/evaluate_humanml3d_table45.py" \
        --mode hybrid --model g1 --dataset-name BONES-SEED-v004 --checkpoint "$ckpt" \
        --manifest "$VAL" --stats-dir "$STATS" --text-cache "$TEXT" --protocol table4 \
        --samples 64 --batch-size 2 --diffusion-steps 10 --cfg-weight-text 2 \
        --cfg-weight-constraint 2 --seed 20260831 --device cuda:0 --panel "$PANEL/table4.jsonl" \
        --protocol-lock "$LOCK" --attention-window-chunks 4 --output "$dir/result.json" \
        --per-sample-output "$dir/per_sample.jsonl" >"$dir/eval.log" 2>&1
    else
      CUDA_VISIBLE_DEVICES=$gpu "$PY" "$ROOT/examples/evaluate_ardy_official_table5.py" \
        --mode hybrid --model g1 --dataset-name BONES-SEED-v004 --checkpoint "$ckpt" \
        --manifest "$VAL" --stats-dir "$STATS" --text-cache "$TEXT" --protocol in_horizon \
        --total-seconds 40 --selection-min-seconds 40 --samples 16 --batch-size 2 \
        --diffusion-steps 10 --cfg-weight-text 2 --cfg-weight-constraint 2 --seed 20260831 \
        --device cuda:0 --panel "$PANEL/40s_in_horizon.jsonl" --protocol-lock "$LOCK" \
        --attention-window-chunks 4 --content-memory-intervention normal --output "$dir/result.json" \
        --per-sample-output "$dir/per_sample.jsonl" --state-diagnostics-output "$dir/state.jsonl" \
        >"$dir/eval.log" 2>&1
    fi
  }
  eval_one 0 state_query_only hybrid table4 & e1=$!
  eval_one 1 persistent_memory_only hybrid table4 & e2=$!
  wait "$e1" || status=1
  wait "$e2" || status=1
  eval_one 0 state_query_only hybrid 40 & e1=$!
  eval_one 1 persistent_memory_only hybrid 40 & e2=$!
  wait "$e1" || status=1
  wait "$e2" || status=1
fi

if (( status == 0 )); then
  "$PY" - "$ROOT" "$OUT" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path

root, out = map(Path, sys.argv[1:])
rows = []
for name in ("native_hybrid", "state_query_only", "persistent_memory_only", "d2rl_50k"):
    for panel in ("table4", "40s"):
        path = out / "eval" / name / panel / "result.json"
        if path.exists():
            data = json.loads(path.read_text())
            rows.append({"model": name, "panel": panel, "result": data})
record = {
    "schema_version": "d2rl_structural_ablation_v1",
    "generated_at": datetime.now().astimezone().isoformat(),
    "dataset": "BONES-SEED-v004",
    "protocol": {"table4": "selection/table4.jsonl", "40s": "selection/40s_in_horizon.jsonl"},
    "models": ["Native Hybrid", "+state_query", "+persistent memory", "D2R-L"],
    "rows": rows,
    "status": "COMPLETE",
    "note": "Only trained variants are reported; Native Hybrid and D2R-L paths are included when available.",
}
json_path = root / "docs/FINAL_STATE_QUERY_MEMORY_ABLATION.json"
json_path.write_text(json.dumps(record, indent=2) + "\n")
lines = [
    "# Final state_query / persistent-memory structural ablation",
    "",
    f"Generated: {record['generated_at']}",
    "",
    "This report uses the same BONES-SEED selection panels and evaluator as the main D2R-L study.",
    "Only completed rows are included; missing baseline rows are not fabricated.",
    "",
    "| Model | Panel | Result artifact |",
    "|---|---|---|",
]
for row in rows:
    lines.append(f"| {row['model']} | {row['panel']} | `{out / 'eval' / row['model'] / row['panel'] / 'result.json'}` |")
lines += ["", "The JSON file preserves all evaluator metrics for paper-table extraction."]
(root / "docs/FINAL_STATE_QUERY_MEMORY_ABLATION.md").write_text("\n".join(lines) + "\n")
PY
fi
date -Is >"$OUT/ABLATION_EVALUATION_COMPLETE" || status=1
exit "$status"
