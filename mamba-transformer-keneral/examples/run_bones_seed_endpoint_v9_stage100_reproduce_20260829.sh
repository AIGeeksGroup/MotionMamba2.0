#!/usr/bin/env bash
set -euo pipefail

ROOT=/data3/zts/MotionMamba2.0/mamba-transformer-keneral
BASE=/data4/zts/result

export STAGE=100
export OUTPUT_ROOT=$BASE/bones_seed_generated_history_endpoint_v9_ft_20260829
export HISTORY_SAMPLING_STRATEGY=deep_v3
export SEQUENTIAL_ROLLOUT_CURRICULUM=generated_deep_v3
export SHORT_WINDOW_PROBABILITY=0.5
export GENERATED_PROBABILITY_START=0.5
export GENERATED_PROBABILITY_1K=0.75
export GENERATED_PROBABILITY_5K=0.75
export GENERATED_MAX_RECOVERY_METERS=4
export GENERATED_BPTT_WINDOWS=4
export GENERATED_MAX_SUPERVISION_POINTS=1
export GENERATED_STATE_CONSISTENCY_WEIGHT=0
export MAMBA_OUTPUT_DECAY_START_WINDOWS=4
export MAMBA_OUTPUT_DECAY_HALF_LIFE_WINDOWS=4
export MAMBA_OUTPUT_DECAY_MIN_SCALE=0
export GENERATED_ROOT_ENDPOINT_LOSS_WEIGHT=5

bash "$ROOT/examples/run_bones_seed_multidepth_v4_ablation_train_20260829.sh"

export SOURCE=$OUTPUT_ROOT
export OUT=$BASE/bones_seed_generated_history_endpoint_v9_val_stage100_20260829
bash "$ROOT/examples/run_bones_seed_multidepth_v4_ablation_val_20260829.sh"

