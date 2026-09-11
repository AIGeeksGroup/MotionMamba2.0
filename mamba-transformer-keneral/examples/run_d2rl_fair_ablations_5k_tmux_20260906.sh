#!/usr/bin/env bash
set -euo pipefail

ROOT=/data3/zts/MotionMamba2.0/mamba-transformer-keneral
PY=/data3/zts/conda_envs/ardy_env/bin/python
BASE=/data4/zts/result
PARENT=$BASE/bones_seed_contract_full50k_20260824_hybrid/hybrid_humanml3d_ddp.pt
TRAIN=$BASE/bones_seed_ardy_train/manifest.jsonl
VAL=$BASE/bones_seed_ardy_val/manifest.jsonl
STATS=$BASE/bones_seed_ardy_train
TEXT=$BASE/bones_seed_text_cache_llm2vec
OUT=$BASE/stage_d2rl_fair_ablations_5k_20260906

export HF_HOME=/data3/zts/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LOCAL_CACHE=true
export PYTHONPATH=$ROOT/src:/data3/zts/MotionMamba2.0/ardy:/data3/zts/packages/state-spaces-mamba:/data3/zts/packages/mamba_op_triton${PYTHONPATH:+:$PYTHONPATH}
[[ ! -e "$OUT" ]] || { echo "refusing to overwrite $OUT" >&2; exit 1; }
mkdir -p "$OUT"

COMMON=(
  --model g1 --mode hybrid --manifest "$TRAIN" --dev-manifest "$VAL" --stats-dir "$STATS"
  --text-cache "$TEXT" --steps 5000 --batch-size 1 --learning-rate 1e-4 --seed 9032026
  --frames 312 --history-probability 0 --constraint-dropout 0.1 --text-dropout 0.1
  --sequential-cache-probability 1 --sequential-rollout-min-windows 6 --sequential-rollout-windows 6
  --sequential-rollout-curriculum d2_post_eviction_v1 --generated-history-probability-start 1
  --generated-history-probability-1k 1 --generated-history-probability-5k 1
  --generated-history-diffusion-steps 10 --generated-history-max-supervision-points 1
  --generated-history-bptt-windows 4 --generated-history-attach-seed-write
  --eval-protocol explicit_history --eval-batches 1 --eval-every 0 --save-every 5000
  --keep-last-checkpoints 1 --no-final-resume-checkpoint --boundary-loss-weight 10
  --root-velocity-loss-weight 50 --constraint-loss-weight 1 --fk-constraint-loss-weight 1
  --foot-skate-loss-weight 50 --train-scope state_memory --attention-window-chunks 4
  --state-summary-dim 128 --use-state-delta --state-conditioning-scale-init 0
  --state-aware-output-mode native_gated_add --init-checkpoint "$PARENT" --log-every 10
)

CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run --standalone --nproc-per-node=4 \
  "$ROOT/examples/train_humanml3d_ardy_ddp.py" "${COMMON[@]}" \
  --token-state-interaction state_query --output-dir "$OUT/state_query_only" \
  >"$OUT/state_query_only.log" 2>&1 &
P1=$!

CUDA_VISIBLE_DEVICES=4,5,6,7 "$PY" -m torch.distributed.run --standalone --nproc-per-node=4 \
  "$ROOT/examples/train_humanml3d_ardy_ddp.py" "${COMMON[@]}" \
  --token-state-interaction naive_gated_add --content-memory-enabled --content-memory-slots 2 \
  --state-aware-output-mode attention_only \
  --content-memory-use-state-delta --content-memory-write-gate-context previous_memory \
  --content-memory-decoder-init-scale 0.02 --content-memory-write-gate-init-scale 0.02 \
  --content-memory-write-gate-bias-init 0 --content-memory-read-scale-init 0 \
  --content-memory-conditioning text --content-memory-normalize-delta --content-memory-normalize-slots \
  --content-memory-max-write-rate 0.25 --output-dir "$OUT/persistent_memory_only" \
  >"$OUT/persistent_memory_only.log" 2>&1 &
P2=$!

status=0
wait "$P1" || status=1
wait "$P2" || status=1
date -Is >"$OUT/ABLATION_TRAINING_COMPLETE"
exit "$status"
