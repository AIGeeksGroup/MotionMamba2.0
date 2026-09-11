#!/usr/bin/env bash
set -euo pipefail
ROOT=/data3/zts/MotionMamba2.0/mamba-transformer-keneral
PY=/data3/zts/conda_envs/ardy_env/bin/python
BASE=/data4/zts/result
OUT=$BASE/stage_d2rl_50k_20260906
RESUME=$BASE/stage_d2rl_bones_long_20260904/step5000/full/checkpoint_step_005000.pt
TRAIN=$BASE/bones_seed_ardy_train/manifest.jsonl
VAL=$BASE/bones_seed_ardy_val/manifest.jsonl
STATS=$BASE/bones_seed_ardy_train
TEXT=$BASE/bones_seed_text_cache_llm2vec
[[ -e "$OUT" ]] && { echo "Refusing to overwrite $OUT" >&2; exit 1; }
mkdir -p "$OUT"
export HF_HOME=/data3/zts/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LOCAL_CACHE=true
export PYTHONPATH=$ROOT/src:/data3/zts/MotionMamba2.0/ardy:/data3/zts/packages/state-spaces-mamba:/data3/zts/packages/mamba_op_triton${PYTHONPATH:+:$PYTHONPATH}
sha256sum "$RESUME" > "$OUT/resume.sha256"
CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run --standalone --nproc-per-node=4 \
  "$ROOT/examples/train_humanml3d_ardy_ddp.py" \
  --model g1 --mode hybrid --manifest "$TRAIN" --dev-manifest "$VAL" --stats-dir "$STATS" --text-cache "$TEXT" \
  --batch-size 1 --learning-rate 1e-4 --override-resume-learning-rate --seed 9032026 --frames 312 \
  --history-probability 0 --constraint-dropout 0.1 --text-dropout 0.1 --sequential-cache-probability 1 \
  --sequential-rollout-min-windows 6 --sequential-rollout-windows 6 --sequential-rollout-curriculum d2_post_eviction_v1 \
  --generated-history-probability-start 1 --generated-history-probability-1k 1 --generated-history-probability-5k 1 \
  --generated-history-diffusion-steps 10 --generated-history-max-supervision-points 1 --generated-history-bptt-windows 4 \
  --generated-history-attach-seed-write --eval-protocol explicit_history --eval-batches 1 --eval-every 5000 \
  --keep-last-checkpoints 2 --save-every 5000 --boundary-loss-weight 10 --root-velocity-loss-weight 50 \
  --constraint-loss-weight 1 --fk-constraint-loss-weight 1 --foot-skate-loss-weight 50 --train-scope state_memory \
  --attention-window-chunks 4 --token-state-interaction state_query --state-aware-output-mode native_gated_add \
  --state-summary-dim 128 --use-state-delta --state-conditioning-scale-init 0 \
  --content-memory-enabled --content-memory-slots 2 --content-memory-use-state-delta \
  --content-memory-write-gate-context previous_memory --content-memory-decoder-init-scale 0.02 \
  --content-memory-write-gate-init-scale 0.02 --content-memory-write-gate-bias-init 0 \
  --content-memory-read-scale-init 0 --content-memory-conditioning text --content-memory-normalize-delta \
  --content-memory-normalize-slots --content-memory-max-write-rate 0.25 --steps 50000 \
  --resume "$RESUME" --output-dir "$OUT" --log-every 10 \
  > "$OUT/train.log" 2>&1
date -Is > "$OUT/TRAINING_COMPLETE"
