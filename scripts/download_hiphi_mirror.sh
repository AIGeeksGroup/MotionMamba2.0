#!/usr/bin/env bash
set -euo pipefail
set +x

: "${HF_TOKEN:?HF_TOKEN must be set; use 'hf auth login' and export it only for this command}"

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data4/zts/.cache/huggingface
export TMPDIR=/data4/zts/tmp

HIPHI_ROOT=${HIPHI_ROOT:-/data4/zts/datasets/HiPHI}
CODE_ROOT=${CODE_ROOT:-/data3/zts/MotionMamba2.0/mamba-transformer-keneral}
ARDY_PYTHON=${ARDY_PYTHON:-/data3/zts/conda_envs/ardy_env/bin/python}
export HIPHI_ROOT
mkdir -p "$HIPHI_ROOT" "$HF_HOME" "$TMPDIR"
mkdir -p /data4/zts/datasets/motion_fusion/logs

python - <<'PY'
import os
from huggingface_hub import snapshot_download

if os.environ.get("HF_ENDPOINT") != "https://hf-mirror.com":
    raise SystemExit("refusing non-mirror Hugging Face endpoint")
common = dict(
    repo_id="noitomrobotics/HiPHI",
    repo_type="dataset",
    local_dir=os.environ.get("HIPHI_ROOT", "/data4/zts/datasets/HiPHI"),
    token=os.environ["HF_TOKEN"],
    endpoint="https://hf-mirror.com",
)
snapshot_download(
    **common,
    allow_patterns=[
        "README.md", "LICENSE.md", "metadata/**", "docs/**",
        "data/archive_manifest.csv", "data/motion_to_part.csv",
    ],
)
snapshot_download(
    **common,
    max_workers=4,
)
PY

"$ARDY_PYTHON" "$CODE_ROOT/examples/data/audit_hiphi_archives.py" \
  --hiphi-root "$HIPHI_ROOT" \
  --readability \
  --output /data4/zts/datasets/motion_fusion/logs/hiphi_archive_audit.json

unset HF_TOKEN
