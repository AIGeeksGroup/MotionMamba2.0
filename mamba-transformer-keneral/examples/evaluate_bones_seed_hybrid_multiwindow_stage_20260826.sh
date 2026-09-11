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
TRAIN=${TRAIN:-$BASE/bones_seed_hybrid_multiwindow_ft_20260826}
OUT=$BASE/bones_seed_hybrid_multiwindow_eval_20260826
STAGE=${STAGE:?set STAGE to the completed training step}
VARIANT=${VARIANT:-v1}
STEM=hybrid_${VARIANT}_stage${STAGE}_40s_in_horizon
mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES=0 "$PY" "$ROOT/examples/evaluate_ardy_official_table5.py" \
  --dataset-name BONES-SEED-v004 --model g1 --mode hybrid \
  --checkpoint "$TRAIN/hybrid_humanml3d_ddp.pt" \
  --manifest "$BASE/bones_seed_ardy_test/manifest.jsonl" \
  --stats-dir "$BASE/bones_seed_ardy_train" \
  --text-cache "$BASE/bones_seed_text_cache_llm2vec" \
  --protocol in_horizon --total-seconds 40 --selection-min-seconds 40 \
  --samples 20 --batch-size 2 --diffusion-steps 10 --cfg-weight-text 2 \
  --cfg-weight-constraint 2 --seed 20260824 --device cuda:0 \
  --output "$OUT/$STEM.json" --artifacts-output "$OUT/$STEM.npz" \
  >"$OUT/$STEM.log" 2>&1

"$PY" - "$STAGE" "$OUT/$STEM.json" "$OUT/$STEM.npz" \
  "$BASE/bones_seed_ardy_retrained_long_table5_20260826/minimal_40s_in_horizon.npz" \
  "$BASE/bones_seed_ardy_retrained_long_table5_20260826/hybrid_40s_in_horizon.npz" \
  "$OUT/hybrid_${VARIANT}_stage${STAGE}_summary.json" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

stage = int(sys.argv[1])
result_path, candidate_path, minimal_path, baseline_path, output_path = map(Path, sys.argv[2:])
result = json.loads(result_path.read_text())
candidate = np.load(candidate_path)
minimal = np.load(minimal_path)
baseline = np.load(baseline_path)
assert result["status"] == "PASS" and result["samples"] == 20
assert np.array_equal(candidate["sample_ids"], minimal["sample_ids"])
assert np.array_equal(candidate["sample_ids"], baseline["sample_ids"])

horizon = int(result["generation_horizon_frames"])

def window_root(artifact):
    error = np.linalg.norm(
        artifact["generated_root"] - artifact["target_root"], axis=-1
    ) * 100.0
    return np.stack(
        [error[:, start:start + horizon].mean(axis=1) for start in range(0, error.shape[1], horizon)],
        axis=1,
    )

minimal_root = window_root(minimal)
baseline_root = window_root(baseline)
candidate_root = window_root(candidate)
x = np.arange(6, candidate_root.shape[1] + 1, dtype=np.float64)
xc = x - x.mean()

def sample_gap_slopes(hybrid_root):
    gap = hybrid_root - minimal_root
    return (gap[:, 5:] * xc[None]).sum(axis=1) / np.square(xc).sum(), gap

baseline_slopes, baseline_gap = sample_gap_slopes(baseline_root)
candidate_slopes, candidate_gap = sample_gap_slopes(candidate_root)
rng = np.random.default_rng(20260826 + stage)
bootstrap = np.asarray([
    rng.choice(candidate_slopes, len(candidate_slopes), replace=True).mean()
    for _ in range(20_000)
])
summary = {
    "status": "PASS",
    "stage": stage,
    "constraint_error_cm_mean": result["constraint_error_cm_mean"],
    "root_ade_cm_mean": result["mobench_root_ade_cm_mean"],
    "baseline_post_window5_gap_slope_cm_per_window": float(baseline_slopes.mean()),
    "candidate_post_window5_gap_slope_cm_per_window": float(candidate_slopes.mean()),
    "candidate_slope_bootstrap_ci95": np.percentile(bootstrap, [2.5, 97.5]).tolist(),
    "slope_improvement_cm_per_window": float(baseline_slopes.mean() - candidate_slopes.mean()),
    "baseline_final_window_gap_cm": float(baseline_gap[:, -1].mean()),
    "candidate_final_window_gap_cm": float(candidate_gap[:, -1].mean()),
    "candidate_root_ade_cm_by_window": candidate_root.mean(axis=0).tolist(),
}
output_path.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
