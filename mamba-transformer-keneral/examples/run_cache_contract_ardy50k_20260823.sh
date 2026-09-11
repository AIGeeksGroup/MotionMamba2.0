#!/usr/bin/env bash
set -euo pipefail
ROOT=/data3/zts/MotionMamba2.0/mamba-transformer-keneral
PY=/data3/zts/conda_envs/ardy_env/bin/python
BASE=/data4/zts/result
COMMON=(--manifest $BASE/humanml3d_ardy_train/manifest.jsonl --dev-manifest $BASE/humanml3d_ardy_val/manifest.jsonl --stats-dir $BASE/humanml3d_ardy_train --text-root /data3/zts/datasets/HumanML3D/HumanML3D/texts --text-cache $BASE/humanml3d_text_cache_llm2vec --steps 50000 --batch-size 8 --learning-rate 5e-5 --seed 123 --frames 80 --history-probability 0.5 --constraint-dropout 0.1 --text-dropout 0.1 --sequential-cache-probability 0.0 --eval-protocol explicit_history --eval-batches 4 --boundary-loss-weight 10 --root-velocity-loss-weight 50 --constraint-loss-weight 1 --fk-constraint-loss-weight 1 --foot-skate-loss-weight 50 --eval-every 1000 --save-every 5000 --keep-last-checkpoints 1)
CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run --standalone --nproc-per-node=4 "$ROOT/examples/train_humanml3d_ardy_ddp.py" "${COMMON[@]}" --mode original --output-dir "$BASE/humanml3d_contractfix_full50k_20260823_original" >"$BASE/humanml3d_contractfix_full50k_20260823_original.log" 2>&1
