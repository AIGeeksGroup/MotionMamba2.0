#!/usr/bin/env bash
set -euo pipefail
ROOT=/data3/zts/MotionMamba2.0/mamba-transformer-keneral
PY=/data3/zts/conda_envs/ardy_env/bin/python
BASE=/data4/zts/result
COMMON=(--manifest $BASE/humanml3d_ardy_train/manifest.jsonl --dev-manifest $BASE/humanml3d_ardy_val/manifest.jsonl --stats-dir $BASE/humanml3d_ardy_train --text-root /data3/zts/datasets/HumanML3D/HumanML3D/texts --text-cache $BASE/humanml3d_text_cache_llm2vec --steps 1000 --batch-size 16 --learning-rate 5e-5 --seed 123 --frames 80 --history-probability 0.5 --constraint-dropout 0.1 --text-dropout 0.1 --sequential-cache-probability 0.5 --eval-protocol both --eval-batches 2 --boundary-loss-weight 10 --root-velocity-loss-weight 50 --constraint-loss-weight 1 --fk-constraint-loss-weight 1 --foot-skate-loss-weight 50 --eval-every 250 --save-every 500 --keep-last-checkpoints 1)
run_one(){ local g=$1 m=$2; CUDA_VISIBLE_DEVICES=$g $PY -m torch.distributed.run --standalone --nproc-per-node=2 $ROOT/examples/train_humanml3d_ardy_ddp.py "${COMMON[@]}" --mode $m --output-dir $BASE/cache_contract_gate1k_$m >$BASE/cache_contract_gate1k_$m.log 2>&1; }
run_one 0,1 minimal &
run_one 2,3 hybrid &
wait
