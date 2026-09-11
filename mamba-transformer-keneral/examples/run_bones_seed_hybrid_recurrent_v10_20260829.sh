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
OUTPUT_ROOT=$BASE/bones_seed_hybrid_recurrent_v10_ft_20260829
OUTPUT=$OUTPUT_ROOT/hybrid
SOURCE=$BASE/bones_seed_hybrid_recurrent_v10_eval_source_20260829
INIT=$BASE/bones_seed_generated_history_endpoint_v9_ft_20260829/hybrid/checkpoint_step_000100.pt
mkdir -p "$OUTPUT" "$SOURCE/minimal" "$SOURCE/hybrid"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$PY" -m torch.distributed.run \
  --standalone --nproc-per-node=8 "$ROOT/examples/train_humanml3d_ardy_ddp.py" \
  --model g1 --mode hybrid --manifest "$BASE/bones_seed_ardy_train/manifest.jsonl" \
  --dev-manifest "$BASE/bones_seed_ardy_val/manifest.jsonl" \
  --stats-dir "$BASE/bones_seed_ardy_train" --text-cache "$BASE/bones_seed_text_cache_llm2vec" \
  --output-dir "$OUTPUT" --steps 100 --batch-size 2 --learning-rate 1e-6 \
  --train-scope recurrent --seed 827321 --frames 988 --history-probability 0 \
  --constraint-dropout 0.1 --text-dropout 0.1 \
  --sequential-cache-probability 0.5 --sequential-rollout-curriculum generated_deep_v3 \
  --history-sampling-strategy deep_v3 --short-window-probability 0.5 \
  --generated-history-probability-start 0.5 --generated-history-probability-1k 0.75 \
  --generated-history-probability-5k 0.75 --generated-history-diffusion-steps 10 \
  --generated-history-cfg-weight-text 2 --generated-history-cfg-weight-constraint 2 \
  --generated-history-max-recovery-meters 4 --generated-history-max-supervision-points 1 \
  --generated-history-bptt-windows 4 --generated-history-root-endpoint-loss-weight 5 \
  --mamba-output-decay-start-windows 4 --mamba-output-decay-half-life-windows 4 \
  --eval-protocol cache_only --eval-batches 1 \
  --boundary-loss-weight 10 --root-velocity-loss-weight 50 \
  --constraint-loss-weight 1 --fk-constraint-loss-weight 1 --foot-skate-loss-weight 50 \
  --eval-every 0 --log-every 10 --save-every 100 --keep-last-checkpoints 2 \
  --init-checkpoint "$INIT"

ln -f "$BASE/bones_seed_generated_history_endpoint_v9_ft_20260829/minimal/checkpoint_step_000100.pt" \
  "$SOURCE/minimal/checkpoint_step_000100.pt"
ln -f "$OUTPUT/checkpoint_step_000100.pt" "$SOURCE/hybrid/checkpoint_step_000100.pt"

SOURCE="$SOURCE" OUT=$BASE/bones_seed_hybrid_recurrent_v10_val_20260829 STAGE=100 \
  MAMBA_OUTPUT_DECAY_START_WINDOWS=4 MAMBA_OUTPUT_DECAY_HALF_LIFE_WINDOWS=4 \
  bash "$ROOT/examples/run_bones_seed_multidepth_v4_ablation_val_20260829.sh"
