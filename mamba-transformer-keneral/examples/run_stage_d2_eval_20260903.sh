#!/usr/bin/env bash
set -euo pipefail

ROOT=/data3/zts/MotionMamba2.0/mamba-transformer-keneral
PY=/data3/zts/conda_envs/ardy_env/bin/python
BASE=/data4/zts/result
D2_STAGE=${D2_STAGE:-100}
D2_FAMILY=${D2_FAMILY:-d2}
if [[ $D2_FAMILY == d2rl ]]; then
  case "$D2_STAGE" in
    5000)
      OUT=$BASE/stage_d2rl_bones_long_20260904/step5000/selection_eval
      FULL=$BASE/stage_d2rl_bones_long_20260904/step5000/full/hybrid_humanml3d_ddp.pt
      NO_DELTA=$BASE/stage_d2rl_bones_long_20260904/step5000/no_delta/hybrid_humanml3d_ddp.pt
      ;;
    *) echo "D2R-L stage must be 5000" >&2; exit 2 ;;
  esac
elif [[ $D2_FAMILY == d2r ]]; then
  case "$D2_STAGE" in
    100)
      OUT=$BASE/stage_d2r_bones_step100_selection_eval_20260903
      FULL=$BASE/stage_d2r_bones_step100_full_v2_20260903/hybrid_humanml3d_ddp.pt
      NO_DELTA=$BASE/stage_d2r_bones_step100_no_delta_v2_20260903/hybrid_humanml3d_ddp.pt
      ;;
    250)
      OUT=$BASE/stage_d2r_bones_step250_selection_eval_20260903
      FULL=$BASE/stage_d2r_bones_step250_full_20260903/hybrid_humanml3d_ddp.pt
      NO_DELTA=$BASE/stage_d2r_bones_step250_no_delta_20260903/hybrid_humanml3d_ddp.pt
      ;;
    *) echo "D2R stage must be 100 or 250" >&2; exit 2 ;;
  esac
else
case "$D2_STAGE" in
  100)
    OUT=$BASE/stage_d2_bones_selection_eval_20260903
    FULL=$BASE/stage_d2_bones_step100_full_v2_20260903/hybrid_humanml3d_ddp.pt
    NO_DELTA=$BASE/stage_d2_bones_step100_no_delta_v2_20260903/hybrid_humanml3d_ddp.pt
    ;;
  250)
    OUT=$BASE/stage_d2_bones_step250_selection_eval_20260903
    FULL=$BASE/stage_d2_bones_step250_full_20260903/hybrid_humanml3d_ddp.pt
    NO_DELTA=$BASE/stage_d2_bones_step250_no_delta_20260903/hybrid_humanml3d_ddp.pt
    ;;
  *)
    echo "D2_STAGE must be 100 or 250" >&2
    exit 2
    ;;
esac
fi
PANELS=$BASE/bones_seed_long_memory_panel_v2_20260831/selection
LOCK=$ROOT/configs/bones_seed_long_memory_protocol_v2.json
MANIFEST=$BASE/bones_seed_ardy_val/manifest.jsonl
STATS=$BASE/bones_seed_ardy_train
TEXT=$BASE/bones_seed_text_cache_llm2vec
NATIVE=$BASE/bones_seed_contract_full50k_20260824_hybrid/hybrid_humanml3d_ddp.pt
MINIMAL=$BASE/bones_seed_contract_full50k_20260824_minimal/minimal_humanml3d_ddp.pt

export HF_HOME=/data3/zts/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LOCAL_CACHE=true
export PYTHONPATH=$ROOT/src:/data3/zts/MotionMamba2.0/ardy:/data3/zts/packages/state-spaces-mamba:/data3/zts/packages/mamba_op_triton${PYTHONPATH:+:$PYTHONPATH}

mkdir -p "$OUT"

checkpoint_for() {
  case "$1" in
    d2_full) echo "$FULL" ;;
    d2_no_delta) echo "$NO_DELTA" ;;
    native_hybrid) echo "$NATIVE" ;;
    minimal_w4) echo "$MINIMAL" ;;
    *) return 1 ;;
  esac
}

mode_for() {
  case "$1" in
    d2_full|d2_no_delta|native_hybrid) echo hybrid ;;
    minimal_w4) echo minimal ;;
    *) return 1 ;;
  esac
}

run_table4() {
  local gpu=$1 variant=$2
  local checkpoint dir
  checkpoint=$(checkpoint_for "$variant")
  dir=$OUT/$variant/table4
  [[ ! -f $dir/result.json ]] || return 0
  [[ ! -e $dir ]] || { echo "refusing to reuse incomplete $dir" >&2; return 1; }
  mkdir -p "$dir"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$ROOT/examples/evaluate_humanml3d_table45.py" \
    --mode "$(mode_for "$variant")" --model g1 --dataset-name BONES-SEED-v004 \
    --checkpoint "$checkpoint" --manifest "$MANIFEST" --stats-dir "$STATS" \
    --text-cache "$TEXT" --protocol table4 --samples 64 --batch-size 2 \
    --diffusion-steps 10 --cfg-weight-text 2 --cfg-weight-constraint 2 \
    --seed 20260831 --device cuda:0 --panel "$PANELS/table4.jsonl" \
    --protocol-lock "$LOCK" --attention-window-chunks 4 \
    --output "$dir/result.json" --per-sample-output "$dir/per_sample.jsonl" \
    >"$dir/eval.log" 2>&1
}

run_long() {
  local gpu=$1 variant=$2 seconds=$3 protocol=$4 intervention=${5:-normal}
  local checkpoint samples selection suffix panel dir
  checkpoint=$(checkpoint_for "$variant")
  samples=16
  selection=$seconds
  if [[ $seconds == 9 ]]; then
    samples=64
  elif [[ $seconds == 20 ]]; then
    selection=40
  fi
  suffix=$variant
  [[ $intervention == normal ]] || suffix=${variant}__${intervention}
  panel=${seconds}s_${protocol}
  dir=$OUT/$suffix/$panel
  [[ ! -f $dir/result.json ]] || return 0
  [[ ! -e $dir ]] || { echo "refusing to reuse incomplete $dir" >&2; return 1; }
  mkdir -p "$dir"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$ROOT/examples/evaluate_ardy_official_table5.py" \
    --mode "$(mode_for "$variant")" --model g1 --dataset-name BONES-SEED-v004 \
    --checkpoint "$checkpoint" --manifest "$MANIFEST" --stats-dir "$STATS" \
    --text-cache "$TEXT" --protocol "$protocol" --total-seconds "$seconds" \
    --selection-min-seconds "$selection" --samples "$samples" --batch-size 2 \
    --diffusion-steps 10 --cfg-weight-text 2 --cfg-weight-constraint 2 \
    --seed 20260831 --device cuda:0 --panel "$PANELS/$panel.jsonl" \
    --protocol-lock "$LOCK" --attention-window-chunks 4 \
    --content-memory-intervention "$intervention" \
    --output "$dir/result.json" --per-sample-output "$dir/per_sample.jsonl" \
    --state-diagnostics-output "$dir/state.jsonl" >"$dir/eval.log" 2>&1
}

run_wave() {
  local status=0
  local pids=()
  while (( $# )); do
    local gpu=$1
    shift
    ( eval "$1" ) & pids+=("$!")
    shift
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || status=1
  done
  (( status == 0 ))
}

# Wave 1: all preregistered short/long health panels.
run_wave \
  0 'run_table4 0 d2_full' \
  1 'run_table4 1 d2_no_delta' \
  2 'run_long 2 d2_full 9 in_horizon' \
  3 'run_long 3 d2_no_delta 9 in_horizon' \
  4 'run_long 4 d2_full 9 out_of_horizon' \
  5 'run_long 5 d2_no_delta 9 out_of_horizon' \
  6 'run_long 6 d2_full 20 in_horizon' \
  7 'run_long 7 d2_no_delta 20 in_horizon'

# Wave 2: 40-second primary plus paired causal interventions on D2-full.
run_wave \
  0 'run_long 0 d2_full 40 in_horizon' \
  1 'run_long 1 d2_no_delta 40 in_horizon' \
  2 'run_long 2 d2_full 40 in_horizon zero_delta' \
  3 'run_long 3 d2_full 40 in_horizon permute_delta' \
  4 'run_long 4 d2_full 40 in_horizon write_off' \
  5 'run_long 5 d2_full 40 in_horizon readout_off' \
  6 'run_long 6 d2_full 40 in_horizon permute_memory'

# Wave 3: healthy 50k controls, not the unrelated Stage-B 100-step variants.
run_wave \
  0 'run_long 0 native_hybrid 20 in_horizon' \
  1 'run_long 1 native_hybrid 40 in_horizon' \
  2 'run_long 2 minimal_w4 20 in_horizon' \
  3 'run_long 3 minimal_w4 40 in_horizon'

"$PY" - "$OUT" "$ROOT" "$FULL" "$NO_DELTA" "$MANIFEST" "$PANELS" "$D2_STAGE" "$D2_FAMILY" <<'PY'
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch

out, root, full, no_delta, manifest, panels = map(Path, sys.argv[1:7])
stage = int(sys.argv[7])
family = sys.argv[8]
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
native = Path("/data4/zts/result/bones_seed_contract_full50k_20260824_hybrid/hybrid_humanml3d_ddp.pt")
minimal = Path("/data4/zts/result/bones_seed_contract_full50k_20260824_minimal/minimal_humanml3d_ddp.pt")
record = {
    "status": "COMPLETE",
    "stage": stage,
    "family": family,
    "git_sha": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip(),
    "git_worktree_dirty": bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True
        ).strip()
    ),
    "source_sha256": {
        str(path.relative_to(root)): sha(path)
        for path in (
            root / "examples/evaluate_ardy_official_table5.py",
            root / "examples/evaluate_humanml3d_table45.py",
            root / "examples/evaluation_checkpoint_config.py",
            root / "src/ardy_adapter/diffusion_memory.py",
            root / "examples/run_stage_d2_eval_20260903.sh",
        )
    },
    "checkpoints": {
        "d2_full": str(full), "d2_no_delta": str(no_delta),
        "native_hybrid": str(native), "minimal_w4": str(minimal),
    },
    "checkpoint_sha256": {
        "d2_full": sha(full), "d2_no_delta": sha(no_delta),
        "native_hybrid": sha(native), "minimal_w4": sha(minimal),
    },
    "manifest": str(manifest),
    "manifest_sha256": sha(manifest),
    "selection_panel_directory": str(panels),
    "seed": 20260831,
    "batch_size": 2,
    "diffusion_steps": 10,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "python": platform.python_version(),
}
(out / "evaluation_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
PY
date -Is >"$OUT/SELECTION_EVALUATION_COMPLETE"
