#!/usr/bin/env bash
set -euo pipefail

ROOT=/data4/zts/datasets/bones_seed
while true; do
  [[ -f "$ROOT/g1.tar.gz" ]] \
    && [[ -f "$ROOT/soma_proportional.tar.gz" ]] \
    && [[ -f "$ROOT/soma_uniform.tar.gz" ]] \
    && break
  sleep 60
done
exec /data3/zts/conda_envs/ardy_env/bin/python \
  /data3/zts/MotionMamba2.0/mamba-transformer-keneral/examples/validate_bones_seed_archives.py
