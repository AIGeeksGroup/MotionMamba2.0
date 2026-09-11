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
OUTPUT_ROOT=${OUTPUT_ROOT:-$BASE/bones_seed_generated_history_multidepth_v4_ablation_ft_20260829}
STAGE=${STAGE:-100}
HISTORY_SAMPLING_STRATEGY=${HISTORY_SAMPLING_STRATEGY:-deep_v3}
SEQUENTIAL_ROLLOUT_CURRICULUM=${SEQUENTIAL_ROLLOUT_CURRICULUM:-generated_deep_v3}
SHORT_WINDOW_PROBABILITY=${SHORT_WINDOW_PROBABILITY:-0.5}
GENERATED_PROBABILITY_START=${GENERATED_PROBABILITY_START:-0.5}
GENERATED_PROBABILITY_1K=${GENERATED_PROBABILITY_1K:-0.75}
GENERATED_PROBABILITY_5K=${GENERATED_PROBABILITY_5K:-0.75}
GENERATED_MAX_RECOVERY_METERS=${GENERATED_MAX_RECOVERY_METERS:-4}
GENERATED_BPTT_WINDOWS=${GENERATED_BPTT_WINDOWS:-0}
GENERATED_MAX_SUPERVISION_POINTS=${GENERATED_MAX_SUPERVISION_POINTS:-3}
GENERATED_STATE_CONSISTENCY_WEIGHT=${GENERATED_STATE_CONSISTENCY_WEIGHT:-0}
MAMBA_OUTPUT_DECAY_START_WINDOWS=${MAMBA_OUTPUT_DECAY_START_WINDOWS:-0}
MAMBA_OUTPUT_DECAY_HALF_LIFE_WINDOWS=${MAMBA_OUTPUT_DECAY_HALF_LIFE_WINDOWS:-0}
MAMBA_OUTPUT_DECAY_MIN_SCALE=${MAMBA_OUTPUT_DECAY_MIN_SCALE:-0}
GENERATED_ROOT_ENDPOINT_LOSS_WEIGHT=${GENERATED_ROOT_ENDPOINT_LOSS_WEIGHT:-0}

initial_for() {
  case "$1" in
    minimal) echo "$BASE/bones_seed_contract_full50k_20260824_minimal/minimal_humanml3d_ddp.pt" ;;
    hybrid) echo "$BASE/bones_seed_contract_full50k_20260824_hybrid/hybrid_humanml3d_ddp.pt" ;;
    *) return 1 ;;
  esac
}

for mode in minimal hybrid; do
  output=$OUTPUT_ROOT/$mode
  mkdir -p "$output"
  if [[ -f $output/checkpoint_step_$(printf '%06d' "$STAGE").pt ]]; then
    continue
  fi
  resume_args=(--init-checkpoint "$(initial_for "$mode")")
  if [[ $STAGE -gt 100 ]]; then
    previous=$(find "$output" -maxdepth 1 -name 'checkpoint_step_*.pt' -print | sort | tail -n 1)
    if [[ -n $previous ]]; then
      resume_args=(--resume "$previous" --override-resume-learning-rate)
    fi
  fi
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$PY" -m torch.distributed.run \
    --standalone --nproc-per-node=8 "$ROOT/examples/train_humanml3d_ardy_ddp.py" \
    --model g1 --mode "$mode" \
    --manifest "$BASE/bones_seed_ardy_train/manifest.jsonl" \
    --dev-manifest "$BASE/bones_seed_ardy_val/manifest.jsonl" \
    --stats-dir "$BASE/bones_seed_ardy_train" \
    --text-cache "$BASE/bones_seed_text_cache_llm2vec" \
    --output-dir "$output" --steps "$STAGE" --batch-size 2 --learning-rate 3e-6 \
    --train-scope denoiser --seed 827321 --frames 988 --history-probability 0 \
    --constraint-dropout 0.1 --text-dropout 0.1 \
    --sequential-cache-probability 0.5 --sequential-rollout-curriculum "$SEQUENTIAL_ROLLOUT_CURRICULUM" \
    --history-sampling-strategy "$HISTORY_SAMPLING_STRATEGY" \
    --short-window-probability "$SHORT_WINDOW_PROBABILITY" \
    --generated-history-probability-start "$GENERATED_PROBABILITY_START" \
    --generated-history-probability-1k "$GENERATED_PROBABILITY_1K" \
    --generated-history-probability-5k "$GENERATED_PROBABILITY_5K" \
    --generated-history-diffusion-steps 10 \
    --generated-history-cfg-weight-text 2 --generated-history-cfg-weight-constraint 2 \
    --generated-history-max-recovery-meters "$GENERATED_MAX_RECOVERY_METERS" \
    --generated-history-depth-ramp-steps 0 \
    --generated-history-max-supervision-points "$GENERATED_MAX_SUPERVISION_POINTS" \
    --generated-history-bptt-windows "$GENERATED_BPTT_WINDOWS" \
    --generated-history-state-consistency-weight "$GENERATED_STATE_CONSISTENCY_WEIGHT" \
    --mamba-output-decay-start-windows "$MAMBA_OUTPUT_DECAY_START_WINDOWS" \
    --mamba-output-decay-half-life-windows "$MAMBA_OUTPUT_DECAY_HALF_LIFE_WINDOWS" \
    --mamba-output-decay-min-scale "$MAMBA_OUTPUT_DECAY_MIN_SCALE" \
    --generated-history-root-endpoint-loss-weight "$GENERATED_ROOT_ENDPOINT_LOSS_WEIGHT" \
    --eval-protocol cache_only --eval-batches 1 \
    --boundary-loss-weight 10 --root-velocity-loss-weight 50 \
    --constraint-loss-weight 1 --fk-constraint-loss-weight 1 --foot-skate-loss-weight 50 \
    --eval-every 0 --log-every 10 --save-every 100 --keep-last-checkpoints 3 \
    "${resume_args[@]}"
done

date -Is >"$OUTPUT_ROOT/STAGE_${STAGE}_COMPLETE"
