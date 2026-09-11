#!/usr/bin/env bash
set -euo pipefail
ROOT=/data3/zts/MotionMamba2.0/mamba-transformer-keneral
PY=/data3/zts/conda_envs/ardy_env/bin/python
BASE=/data4/zts/result
PARENT=$BASE/bones_seed_contract_full50k_20260824_hybrid/hybrid_humanml3d_ddp.pt
OUT=$BASE/stage_d2rl_fair_ablations_5k_20260906/persistent_memory_only_v2
export HF_HOME=/data3/zts/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LOCAL_CACHE=true
export PYTHONPATH=$ROOT/src:/data3/zts/MotionMamba2.0/ardy:/data3/zts/packages/state-spaces-mamba:/data3/zts/packages/mamba_op_triton${PYTHONPATH:+:$PYTHONPATH}
[[ ! -e "$OUT" ]] || { echo "refusing to overwrite $OUT" >&2; exit 1; }
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=4,5,6,7 "$PY" -m torch.distributed.run --standalone --nproc-per-node=4 \
  "$ROOT/examples/train_humanml3d_ardy_ddp.py" \
  --model g1 --mode hybrid --manifest "$BASE/bones_seed_ardy_train/manifest.jsonl" \
  --dev-manifest "$BASE/bones_seed_ardy_val/manifest.jsonl" --stats-dir "$BASE/bones_seed_ardy_train" \
  --text-cache "$BASE/bones_seed_text_cache_llm2vec" --steps 5000 --batch-size 1 --learning-rate 1e-4 \
  --seed 9032026 --frames 312 --history-probability 0 --constraint-dropout 0.1 --text-dropout 0.1 \
  --sequential-cache-probability 1 --sequential-rollout-min-windows 6 --sequential-rollout-windows 6 \
  --sequential-rollout-curriculum d2_post_eviction_v1 --generated-history-probability-start 1 \
  --generated-history-probability-1k 1 --generated-history-probability-5k 1 \
  --generated-history-diffusion-steps 10 --generated-history-max-supervision-points 1 \
  --generated-history-bptt-windows 4 --generated-history-attach-seed-write --eval-protocol explicit_history \
  --eval-batches 1 --eval-every 0 --save-every 5000 --keep-last-checkpoints 1 --no-final-resume-checkpoint \
  --boundary-loss-weight 10 --root-velocity-loss-weight 50 --constraint-loss-weight 1 \
  --fk-constraint-loss-weight 1 --foot-skate-loss-weight 50 --train-scope state_memory \
  --attention-window-chunks 4 --token-state-interaction naive_gated_add --state-aware-output-mode attention_only \
  --state-summary-dim 128 --use-state-delta --state-conditioning-scale-init 0 --content-memory-enabled \
  --content-memory-slots 2 --content-memory-use-state-delta --content-memory-write-gate-context previous_memory \
  --content-memory-decoder-init-scale 0.02 --content-memory-write-gate-init-scale 0.02 \
  --content-memory-write-gate-bias-init 0 --content-memory-read-scale-init 0 --content-memory-conditioning text \
  --content-memory-normalize-delta --content-memory-normalize-slots --content-memory-max-write-rate 0.25 \
  --init-checkpoint "$PARENT" --log-every 10 >"$OUT/train.log" 2>&1
date -Is >"$OUT/TRAINING_COMPLETE"
