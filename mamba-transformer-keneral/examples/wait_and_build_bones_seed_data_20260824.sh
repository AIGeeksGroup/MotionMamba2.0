#!/usr/bin/env bash
set -euo pipefail

ROOT=/data4/zts/datasets/bones_seed
VALIDATION="$ROOT/archive_validation.json"
LOG_PREFIX="$(date -Is)"
echo "$LOG_PREFIX waiting_for_all_archives_validation"
while true; do
  if [[ -f "$VALIDATION" ]] && /data3/zts/conda_envs/ardy_env/bin/python -c \
    'import json, sys; sys.exit(json.load(open(sys.argv[1]))["status"] != "PASS")' \
    "$VALIDATION"; then
    break
  fi
  sleep 60
done
echo "$(date -Is) all_archives_validated"
exec /data3/zts/MotionMamba2.0/mamba-transformer-keneral/examples/run_bones_seed_data_build_20260824.sh
