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
OUTPUT=${OUTPUT:-$BASE/bones_seed_hybrid_multiwindow_ft_20260826}
INITIAL=$BASE/bones_seed_contract_full50k_20260824_hybrid/hybrid_humanml3d_ddp.pt
TRAIN=$BASE/bones_seed_ardy_train
VAL=$BASE/bones_seed_ardy_val
TEXT=$BASE/bones_seed_text_cache_llm2vec
TARGET_STEPS=${TARGET_STEPS:-100}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
TRAIN_SCOPE=${TRAIN_SCOPE:-denoiser}
LONG_ROLLOUT_PROBABILITY=${LONG_ROLLOUT_PROBABILITY:-1.0}
LOG=$OUTPUT/train.stdout.log

mkdir -p "$OUTPUT"

latest_checkpoint() {
  find "$OUTPUT" -maxdepth 1 -type f -name 'checkpoint_step_*.pt' -print | sort | tail -n 1
}

resume_args=()
latest=$(latest_checkpoint)
if [[ -n "$latest" ]]; then
  resume_args=(--resume "$latest" --override-resume-learning-rate)
else
  resume_args=(--init-checkpoint "$INITIAL")
fi

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$PY" -m torch.distributed.run \
  --standalone --nproc-per-node=8 "$ROOT/examples/train_humanml3d_ardy_ddp.py" \
  --model g1 --mode hybrid --manifest "$TRAIN/manifest.jsonl" \
  --dev-manifest "$VAL/manifest.jsonl" --stats-dir "$TRAIN" --text-cache "$TEXT" \
  --output-dir "$OUTPUT" --steps "$TARGET_STEPS" --batch-size 4 \
  --learning-rate "$LEARNING_RATE" --train-scope "$TRAIN_SCOPE" \
  --seed 321 --frames 520 --history-probability 0 --constraint-dropout 0.1 \
  --text-dropout 0.1 --sequential-cache-probability 1 \
  --sequential-rollout-min-windows 6 --sequential-rollout-windows 10 \
  --sequential-long-rollout-probability "$LONG_ROLLOUT_PROBABILITY" \
  --eval-protocol cache_only --eval-batches 1 --boundary-loss-weight 10 \
  --root-velocity-loss-weight 50 --constraint-loss-weight 1 \
  --fk-constraint-loss-weight 1 --foot-skate-loss-weight 50 \
  --eval-every 50 --save-every 100 --keep-last-checkpoints 10 \
  "${resume_args[@]}" >>"$LOG" 2>&1

"$PY" - "$OUTPUT/metrics.json" "$OUTPUT/hybrid_humanml3d_ddp.pt" "$TARGET_STEPS" <<'PY'
import json
import sys
from pathlib import Path

metrics_path, checkpoint, steps = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
metrics = json.loads(metrics_path.read_text())
counts = {int(key): int(value) for key, value in metrics["rollout_window_counts"].items()}
valid = (
    checkpoint.is_file()
    and metrics["steps"] == steps
    and metrics["world_size"] == 8
    and metrics["global_batch_size"] == 32
    and metrics["all_gradients_finite"] is True
    and metrics["final_eval_loss"] < metrics["initial_eval_loss"]
    and set(counts).issubset({2, *range(6, 11)})
    and sum(counts.values()) == steps
)
if not valid:
    raise SystemExit(f"multiwindow training gate failed: {metrics}")
print(
    f"multiwindow_stage_pass steps={steps} initial_eval={metrics['initial_eval_loss']:.6f} "
    f"final_eval={metrics['final_eval_loss']:.6f} counts={counts}"
)
PY
