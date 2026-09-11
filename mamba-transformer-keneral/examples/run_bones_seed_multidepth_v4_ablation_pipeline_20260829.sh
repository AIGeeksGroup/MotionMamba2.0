#!/usr/bin/env bash
set -euo pipefail

ROOT=/data3/zts/MotionMamba2.0/mamba-transformer-keneral
STAGE=${STAGE:-100}
export STAGE

bash "$ROOT/examples/run_bones_seed_multidepth_v4_ablation_train_20260829.sh"
bash "$ROOT/examples/run_bones_seed_multidepth_v4_ablation_val_20260829.sh"
