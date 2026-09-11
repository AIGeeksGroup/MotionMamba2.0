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
TRAIN_ROOT=$BASE/bones_seed_generated_history_v2_earlystop900_ft_20260828
SOURCE_EVAL=$BASE/bones_seed_generated_history_v2_earlystop900_eval_20260828
OUT=$BASE/bones_seed_generated_history_v2_earlystop900_seed20260829_n64
TRAIN=$BASE/bones_seed_ardy_train
TEST_SOURCE=$BASE/bones_seed_ardy_test/manifest.jsonl
TEST=$OUT/manifest_excluding_seed20260824_long_samples.jsonl
TEXT=$BASE/bones_seed_text_cache_llm2vec
REPORT=$ROOT/docs/BONES_SEED_GENERATED_HISTORY_V2_EARLYSTOP900_SEED20260829_N64.md
SEED=20260829
SAMPLES=64
mkdir -p "$OUT"

"$PY" - "$SOURCE_EVAL" "$TEST_SOURCE" "$TEST" <<'PY'
import json, sys
from pathlib import Path
source_eval, source_manifest, output = map(Path, sys.argv[1:])
excluded = set()
for mode in ("minimal", "hybrid"):
    for seconds in (20, 40):
        row = json.loads((source_eval / f"{mode}_{seconds}s_in_horizon.json").read_text())
        excluded.update(row["sample_ids"])
kept = []
for line in source_manifest.read_text().splitlines():
    row = json.loads(line)
    if row["id"] not in excluded:
        kept.append(line)
output.write_text("\n".join(kept) + "\n")
(output.parent / "excluded_sample_ids.json").write_text(
    json.dumps(sorted(excluded), indent=2) + "\n"
)
print(f"excluded={len(excluded)} kept={len(kept)}")
PY

result_pass() {
  local path=$1 expected=$2
  [[ -f $path ]] || return 1
  "$PY" - "$path" "$expected" <<'PY'
import json, sys
from pathlib import Path
row = json.loads(Path(sys.argv[1]).read_text())
raise SystemExit(0 if row.get("status") == "PASS" and row.get("samples") == int(sys.argv[2]) else 1)
PY
}

run_table4() {
  local stem=hybrid_table4 checkpoint=$TRAIN_ROOT/hybrid/checkpoint_step_000900.pt
  result_pass "$OUT/$stem.json" "$SAMPLES" && [[ -f $OUT/$stem.npz ]] && return
  CUDA_VISIBLE_DEVICES=0 "$PY" "$ROOT/examples/evaluate_humanml3d_table45.py" \
    --dataset-name BONES-SEED-v004 --model g1 --mode hybrid --checkpoint "$checkpoint" \
    --manifest "$TEST" --stats-dir "$TRAIN" --text-cache "$TEXT" --protocol table4 \
    --samples "$SAMPLES" --batch-size 4 --diffusion-steps 10 --cfg-weight-text 2 \
    --cfg-weight-constraint 2 --seed "$SEED" --device cuda:0 --output "$OUT/$stem.json" \
    --artifacts-output "$OUT/$stem.npz" >"$OUT/$stem.log" 2>&1
}

run_duration() {
  local gpu=$1 mode=$2 seconds=$3
  local stem=${mode}_${seconds}s_in_horizon
  local checkpoint=$TRAIN_ROOT/$mode/checkpoint_step_000900.pt diag=()
  if result_pass "$OUT/$stem.json" "$SAMPLES" && [[ -f $OUT/$stem.npz ]]; then
    if [[ $mode != hybrid || $seconds != 40 || -f $OUT/$stem.state.jsonl ]]; then return; fi
  fi
  if [[ $mode == hybrid && $seconds == 40 ]]; then
    diag=(--state-diagnostics-output "$OUT/$stem.state.jsonl")
  fi
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$ROOT/examples/evaluate_ardy_official_table5.py" \
    --dataset-name BONES-SEED-v004 --model g1 --mode "$mode" --checkpoint "$checkpoint" \
    --manifest "$TEST" --stats-dir "$TRAIN" --text-cache "$TEXT" --protocol in_horizon \
    --total-seconds "$seconds" --selection-min-seconds 40 --samples "$SAMPLES" \
    --batch-size 2 --diffusion-steps 10 --cfg-weight-text 2 --cfg-weight-constraint 2 \
    --seed "$SEED" --device cuda:0 --output "$OUT/$stem.json" \
    --artifacts-output "$OUT/$stem.npz" "${diag[@]}" >"$OUT/$stem.log" 2>&1
}

run_9s() {
  local gpu=$1 protocol=$2
  local stem=hybrid_9s_${protocol}
  local checkpoint=$TRAIN_ROOT/hybrid/checkpoint_step_000900.pt
  result_pass "$OUT/$stem.json" "$SAMPLES" && [[ -f $OUT/$stem.npz ]] && return
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$ROOT/examples/evaluate_ardy_official_table5.py" \
    --dataset-name BONES-SEED-v004 --model g1 --mode hybrid --checkpoint "$checkpoint" \
    --manifest "$TEST" --stats-dir "$TRAIN" --text-cache "$TEXT" --protocol "$protocol" \
    --total-seconds 9 --selection-min-seconds 9 --samples "$SAMPLES" --batch-size 2 \
    --diffusion-steps 10 --cfg-weight-text 2 --cfg-weight-constraint 2 --seed "$SEED" \
    --device cuda:0 --output "$OUT/$stem.json" --artifacts-output "$OUT/$stem.npz" \
    >"$OUT/$stem.log" 2>&1
}

pids=()
run_table4 & pids+=("$!")
run_9s 1 in_horizon & pids+=("$!")
run_9s 2 out_of_horizon & pids+=("$!")
run_duration 3 minimal 20 & pids+=("$!")
run_duration 4 hybrid 20 & pids+=("$!")
run_duration 6 minimal 40 & pids+=("$!")
run_duration 7 hybrid 40 & pids+=("$!")
for pid in "${pids[@]}"; do wait "$pid"; done
date -Is >"$OUT/EVALUATION_COMPLETE"

gate_exit=0
"$PY" "$ROOT/examples/check_bones_seed_generated_history_gate.py" \
  --eval-root "$OUT" --training-root "$TRAIN_ROOT" --stage 900 \
  --output "$OUT/gate.json" || gate_exit=$?

"$PY" - "$OUT" "$REPORT" "$gate_exit" <<'PY'
import json, sys
from pathlib import Path
out, report, gate_exit = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
gate = json.loads((out / "gate.json").read_text())
excluded = set(json.loads((out / "excluded_sample_ids.json").read_text()))
rows = {}
for seconds in (20, 40):
    for mode in ("minimal", "hybrid"):
        rows[(seconds, mode)] = json.loads(
            (out / f"{mode}_{seconds}s_in_horizon.json").read_text()
        )
used = set(rows[(20, "hybrid")]["sample_ids"]) | set(rows[(40, "hybrid")]["sample_ids"])
overlap = sorted(used & excluded)
lines = [
    "# BONES-SEED Step-900 Independent-Seed Confirmation",
    "",
    "This confirmation evaluates the selected GH-v2 step-900 checkpoints with seed 20260829 on 64 long-horizon samples after excluding every long-horizon sample used by the seed-20260824 selection sweep.",
    "",
    f"- Gate status: **{gate['status']}** (checker exit {gate_exit})",
    f"- Excluded prior samples: {len(excluded)}",
    f"- Overlap with prior long-horizon samples: {len(overlap)}",
    f"- Training healthy/matched: {gate['criteria']['training_healthy']}/{gate['criteria']['matched_draws']}",
    f"- Short retention passed: {gate['criteria']['short_retention']}",
    f"- State finite/no explosion: {gate['criteria']['state_finite']}/{gate['criteria']['no_state_norm_explosion']}",
    "",
    "| Duration | Method | Constraint (cm) | Root ADE (cm) | Root slope | Constraint slope |",
    "|---:|---|---:|---:|---:|---:|",
]
for seconds in (20, 40):
    for mode, label in (("minimal", "Minimal"), ("hybrid", "Hybrid")):
        row = rows[(seconds, mode)]
        lines.append(
            f"| {seconds} s | {label} | {row['constraint_error_cm_mean']:.2f} | "
            f"{row['mobench_root_ade_cm_mean']:.2f} | "
            f"{row['post_window5_root_ade_slope_cm_per_window']:.3f} | "
            f"{row['post_window5_constraint_slope_cm_per_window']:.3f} |"
        )
lines.extend(["", "| Duration | Hybrid-Minimal paired Root-slope gap | 95% CI | Hybrid-better fraction |", "|---:|---:|---:|---:|"])
for seconds in (20, 40):
    paired = gate["long_horizon"][f"{seconds}s"]["paired_root_slope_gap"]
    lines.append(
        f"| {seconds} s | {paired['mean_cm_per_window']:.3f} | "
        f"[{paired['ci95_cm_per_window'][0]:.3f}, {paired['ci95_cm_per_window'][1]:.3f}] | "
        f"{paired['hybrid_better_fraction']:.2f} |"
    )
lines.extend(["", "Negative Hybrid-Minimal gaps favor Hybrid. This file is independent of the exploratory checkpoint-sweep report.", ""])
report.parent.mkdir(parents=True, exist_ok=True)
report.write_text("\n".join(lines))
print(report)
PY

date -Is >"$OUT/CONFIRMATION_COMPLETE"
