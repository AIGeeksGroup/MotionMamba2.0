#!/usr/bin/env bash
set -euo pipefail

# Restore the recorded parent checkpoint/cache before running. Their historical
# paths and SHA-256 are in configs/stage_c_gate_20260902.json.
python examples/train_ardy_c1_cached.py \
  --ardy-root /path/to/ardy \
  --checkpoints-dir /path/to/ardy_checkpoints \
  --minimal-checkpoint /path/to/sequence_minimal.pt \
  --train-manifest /path/to/manifest_train.json \
  --eval-manifest /path/to/manifest_eval.json \
  --output-dir /path/to/stage_c_state_delta_100step \
  --interaction state_query --use-state-delta \
  --max-steps 100 --max-train-trajectories 1 --max-eval-trajectories 1 \
  --fixed-case --learning-rate 1e-3 --new-parameter-lr-multiplier 5 \
  --seed 61 --precision bf16
