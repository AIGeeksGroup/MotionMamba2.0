#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src:/data3/zts/packages/state-spaces-mamba:/data3/zts/packages/mamba_op_triton \
  /data3/zts/conda_envs/ardy_env/bin/python -m pytest -q \
  tests/test_content_memory.py \
  tests/test_causal_memory_benchmark.py \
  tests/test_sequence_kernel.py \
  tests/test_sequence_kernel_mamba.py \
  tests/test_ardy_dit_sequence.py \
  tests/test_cache_history_contract.py \
  tests/test_differentiable_rollout.py \
  tests/test_long_memory_protocol.py
