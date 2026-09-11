#!/usr/bin/env bash
set -euo pipefail
ROOT=/data3/zts/MotionMamba2.0/mamba-transformer-keneral
PY=/data3/zts/conda_envs/ardy_env/bin/python
BASE=/data4/zts/result
OUT=${OUT:-$BASE/bones_seed_long_sequence_scaling_v2_20260831}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3} "$PY" "$ROOT/benchmarks/benchmark_ardy_long_sequence_scaling.py" \
  --output-dir "$OUT" --device cuda:0 --windows 1 2 4 8 16 32 \
  --warmup 3 --repeats 10 \
  --ardy-checkpoint "$BASE/bones_seed_ardy_retrain_lr1e5_20260826_original/original_humanml3d_ddp.pt" \
  --minimal-checkpoint "$BASE/bones_seed_contract_full50k_20260824_minimal/minimal_humanml3d_ddp.pt" \
  --hybrid-checkpoint "$BASE/bones_seed_contract_full50k_20260824_hybrid/hybrid_humanml3d_ddp.pt"
