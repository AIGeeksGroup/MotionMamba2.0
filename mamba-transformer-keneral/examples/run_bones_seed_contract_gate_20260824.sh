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
TRAIN=$BASE/bones_seed_ardy_train
VAL=$BASE/bones_seed_ardy_val
TEXT=$BASE/bones_seed_text_cache_llm2vec
STEPS=${STEPS:-1000}
TAG=${TAG:-gate1k}
PARENT_TAG=${PARENT_TAG:-}

latest_checkpoint() {
  local output=$1
  [[ -d "$output" ]] || return 0
  find "$output" -maxdepth 1 -type f -name 'checkpoint_step_*.pt' -print 2>/dev/null \
    | sort | tail -n 1
}

resume_checkpoint() {
  local output=$1 mode=$2 latest
  latest=$(latest_checkpoint "$output")
  if [[ -z "$latest" ]] && [[ -n "$PARENT_TAG" ]]; then
    latest=$(latest_checkpoint "$BASE/bones_seed_${PARENT_TAG}_${mode}")
  fi
  echo "$latest"
}

training_pass() {
  local output=$1 mode=$2
  "$PY" - "$output/metrics.json" "$output/${mode}_humanml3d_ddp.pt" "$mode" "$STEPS" <<'PY'
import json
import sys
from pathlib import Path

metrics_path, checkpoint, mode, steps = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])
if not metrics_path.is_file() or not checkpoint.is_file() or checkpoint.stat().st_size == 0:
    raise SystemExit(1)
metrics = json.loads(metrics_path.read_text())
valid = (
    metrics.get("status") == "PASS"
    and metrics.get("mode") == mode
    and metrics.get("model") == "g1"
    and metrics.get("steps") == steps
    and metrics.get("world_size") == 4
    and metrics.get("global_batch_size") == 32
    and metrics.get("all_gradients_finite") is True
)
raise SystemExit(0 if valid else 1)
PY
}

archive_unresumable_log() {
  local output=$1
  if [[ -f "$output/train.jsonl" ]] && [[ -z "$(latest_checkpoint "$output")" ]]; then
    mv "$output/train.jsonl" \
      "$output/train.failed.$(date +%Y%m%dT%H%M%S).jsonl"
  fi
}

COMMON=(
  --model g1
  --manifest "$TRAIN/manifest.jsonl"
  --dev-manifest "$VAL/manifest.jsonl"
  --stats-dir "$TRAIN"
  --text-cache "$TEXT"
  --steps "$STEPS"
  --batch-size 8
  --learning-rate 5e-5
  --seed 123
  --frames 104
  --history-probability 0.5
  --constraint-dropout 0.1
  --text-dropout 0.1
  --sequential-cache-probability 0.5
  --eval-protocol both
  --eval-batches 4
  --boundary-loss-weight 10
  --root-velocity-loss-weight 50
  --constraint-loss-weight 1
  --fk-constraint-loss-weight 1
  --foot-skate-loss-weight 50
  --eval-every 500
  --save-every 1000
  --keep-last-checkpoints 1
)

MINIMAL_OUT=$BASE/bones_seed_${TAG}_minimal
HYBRID_OUT=$BASE/bones_seed_${TAG}_hybrid
ORIGINAL_OUT=$BASE/bones_seed_${TAG}_original
MINIMAL_RESUME=()
HYBRID_RESUME=()
ORIGINAL_RESUME=()
latest=$(resume_checkpoint "$MINIMAL_OUT" minimal)
[[ -n "$latest" ]] && MINIMAL_RESUME=(--resume "$latest")
latest=$(resume_checkpoint "$HYBRID_OUT" hybrid)
[[ -n "$latest" ]] && HYBRID_RESUME=(--resume "$latest")
latest=$(resume_checkpoint "$ORIGINAL_OUT" original)
[[ -n "$latest" ]] && ORIGINAL_RESUME=(--resume "$latest")

minimal_pid=
hybrid_pid=
minimal_complete=false
if training_pass "$MINIMAL_OUT" minimal; then
  minimal_complete=true
  echo "skip_completed mode=minimal tag=$TAG steps=$STEPS"
else
  archive_unresumable_log "$MINIMAL_OUT"
  CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run --standalone --nproc-per-node=4 \
    "$ROOT/examples/train_humanml3d_ardy_ddp.py" "${COMMON[@]}" --mode minimal \
    --output-dir "$MINIMAL_OUT" "${MINIMAL_RESUME[@]}" \
    >"$BASE/bones_seed_${TAG}_minimal.log" 2>&1 &
  minimal_pid=$!
fi

if training_pass "$HYBRID_OUT" hybrid; then
  echo "skip_completed mode=hybrid tag=$TAG steps=$STEPS"
else
  archive_unresumable_log "$HYBRID_OUT"
  hybrid_gpus=4,5,6,7
  [[ "$minimal_complete" == true ]] && hybrid_gpus=0,1,2,3
  CUDA_VISIBLE_DEVICES="$hybrid_gpus" "$PY" -m torch.distributed.run --standalone --nproc-per-node=4 \
    "$ROOT/examples/train_humanml3d_ardy_ddp.py" "${COMMON[@]}" --mode hybrid \
    --output-dir "$HYBRID_OUT" "${HYBRID_RESUME[@]}" \
    >"$BASE/bones_seed_${TAG}_hybrid.log" 2>&1 &
  hybrid_pid=$!
fi

[[ -z "$minimal_pid" ]] || wait "$minimal_pid"
[[ -z "$hybrid_pid" ]] || wait "$hybrid_pid"

if training_pass "$ORIGINAL_OUT" original; then
  echo "skip_completed mode=original tag=$TAG steps=$STEPS"
else
  archive_unresumable_log "$ORIGINAL_OUT"
  CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run --standalone --nproc-per-node=4 \
    "$ROOT/examples/train_humanml3d_ardy_ddp.py" "${COMMON[@]}" --mode original \
    --sequential-cache-probability 0 --eval-protocol explicit_history \
    --output-dir "$ORIGINAL_OUT" "${ORIGINAL_RESUME[@]}" \
    >"$BASE/bones_seed_${TAG}_original.log" 2>&1
fi
