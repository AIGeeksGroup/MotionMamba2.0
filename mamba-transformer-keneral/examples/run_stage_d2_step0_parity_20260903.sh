#!/usr/bin/env bash
set -euo pipefail

ROOT=/data3/zts/MotionMamba2.0/mamba-transformer-keneral
PY=/data3/zts/conda_envs/ardy_env/bin/python
BASE=/data4/zts/result
GPU=${D2_PARITY_GPU:-0}

export HF_HOME=/data3/zts/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LOCAL_CACHE=true
export PYTHONPATH=$ROOT/src:/data3/zts/MotionMamba2.0/ardy:/data3/zts/packages/state-spaces-mamba:/data3/zts/packages/mamba_op_triton${PYTHONPATH:+:$PYTHONPATH}

CUDA_VISIBLE_DEVICES=$GPU "$PY" "$ROOT/examples/evaluate_stage_d2_step0_native_parity.py" \
  --checkpoint "$BASE/bones_seed_contract_full50k_20260824_hybrid/hybrid_humanml3d_ddp.pt" \
  --manifest "$BASE/bones_seed_ardy_val/manifest.jsonl" \
  --stats-dir "$BASE/bones_seed_ardy_train" \
  --text-cache "$BASE/bones_seed_text_cache_llm2vec" \
  --output-dir "$BASE/stage_d2_step0_native_parity_20260903" \
  --device cuda:0 --seed 9032026 --sample-index 0 --model g1
