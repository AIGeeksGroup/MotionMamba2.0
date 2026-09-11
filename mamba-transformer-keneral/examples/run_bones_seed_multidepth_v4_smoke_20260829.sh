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
MODE=${MODE:-hybrid}
OUTPUT=${OUTPUT:-$BASE/bones_seed_generated_history_multidepth_v4_smoke_20260829/$MODE}
BPTT_WINDOWS=${BPTT_WINDOWS:-0}
STATE_CONSISTENCY_WEIGHT=${STATE_CONSISTENCY_WEIGHT:-0}
SEED=${SEED:-827342}
MAMBA_OUTPUT_DECAY_START_WINDOWS=${MAMBA_OUTPUT_DECAY_START_WINDOWS:-0}
MAMBA_OUTPUT_DECAY_HALF_LIFE_WINDOWS=${MAMBA_OUTPUT_DECAY_HALF_LIFE_WINDOWS:-0}
GENERATED_ROOT_ENDPOINT_LOSS_WEIGHT=${GENERATED_ROOT_ENDPOINT_LOSS_WEIGHT:-0}
case "$MODE" in
  minimal) INIT=$BASE/bones_seed_contract_full50k_20260824_minimal/minimal_humanml3d_ddp.pt ;;
  hybrid) INIT=$BASE/bones_seed_contract_full50k_20260824_hybrid/hybrid_humanml3d_ddp.pt ;;
  *) echo "smoke supports minimal or hybrid" >&2; exit 2 ;;
esac

mkdir -p "$OUTPUT"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$PY" -m torch.distributed.run \
  --standalone --nproc-per-node=8 "$ROOT/examples/train_humanml3d_ardy_ddp.py" \
  --model g1 --mode "$MODE" \
  --manifest "$BASE/bones_seed_ardy_train/manifest.jsonl" \
  --dev-manifest "$BASE/bones_seed_ardy_val/manifest.jsonl" \
  --stats-dir "$BASE/bones_seed_ardy_train" \
  --text-cache "$BASE/bones_seed_text_cache_llm2vec" \
  --output-dir "$OUTPUT" --steps 1 --batch-size 1 --learning-rate 3e-6 \
  --train-scope denoiser --seed "$SEED" --frames 988 --history-probability 0 \
  --constraint-dropout 0.1 --text-dropout 0.1 \
  --sequential-cache-probability 1 --sequential-rollout-curriculum generated_deep_v3 \
  --history-sampling-strategy deep_v3 --short-window-probability 0 \
  --generated-history-probability-start 1 --generated-history-probability-1k 1 \
  --generated-history-probability-5k 1 --generated-history-diffusion-steps 2 \
  --generated-history-cfg-weight-text 2 --generated-history-cfg-weight-constraint 2 \
  --generated-history-max-recovery-meters 4 \
  --generated-history-max-supervision-points 3 \
  --generated-history-bptt-windows "$BPTT_WINDOWS" \
  --generated-history-state-consistency-weight "$STATE_CONSISTENCY_WEIGHT" \
  --mamba-output-decay-start-windows "$MAMBA_OUTPUT_DECAY_START_WINDOWS" \
  --mamba-output-decay-half-life-windows "$MAMBA_OUTPUT_DECAY_HALF_LIFE_WINDOWS" \
  --generated-history-root-endpoint-loss-weight "$GENERATED_ROOT_ENDPOINT_LOSS_WEIGHT" \
  --eval-protocol cache_only --eval-batches 1 \
  --boundary-loss-weight 10 --root-velocity-loss-weight 50 \
  --constraint-loss-weight 1 --fk-constraint-loss-weight 1 --foot-skate-loss-weight 50 \
  --eval-every 0 --save-every 0 --no-final-resume-checkpoint \
  --init-checkpoint "$INIT"
