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
OUT=$BASE/bones_seed_generated_history_v2_earlystop900_val_seed20260829
TRAIN=$BASE/bones_seed_ardy_train
MANIFEST=$BASE/bones_seed_ardy_val/manifest.jsonl
TEXT=$BASE/bones_seed_text_cache_llm2vec
CURRENT=$BASE/bones_seed_generated_history_v2_earlystop900_ft_20260828
HEALTHY_HYBRID=$BASE/bones_seed_contract_full50k_20260824_hybrid/hybrid_humanml3d_ddp.pt
REPORT=$ROOT/docs/BONES_SEED_GENERATED_HISTORY_V2_EARLYSTOP900_VAL_CONFIRMATION_20260828.md
SEED=20260829
mkdir -p "$OUT"

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
  local gpu=$1 tag=$2 checkpoint=$3
  local stem=${tag}_table4
  result_pass "$OUT/$stem.json" 64 && [[ -f $OUT/$stem.npz ]] && return
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$ROOT/examples/evaluate_humanml3d_table45.py" \
    --dataset-name BONES-SEED-v004-val --model g1 --mode hybrid --checkpoint "$checkpoint" \
    --manifest "$MANIFEST" --stats-dir "$TRAIN" --text-cache "$TEXT" --protocol table4 \
    --samples 64 --batch-size 4 --diffusion-steps 10 --cfg-weight-text 2 \
    --cfg-weight-constraint 2 --seed "$SEED" --device cuda:0 --output "$OUT/$stem.json" \
    --artifacts-output "$OUT/$stem.npz" >"$OUT/$stem.log" 2>&1
}

run_9s() {
  local gpu=$1 tag=$2 checkpoint=$3 protocol=$4
  local stem=${tag}_9s_${protocol}
  result_pass "$OUT/$stem.json" 64 && [[ -f $OUT/$stem.npz ]] && return
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$ROOT/examples/evaluate_ardy_official_table5.py" \
    --dataset-name BONES-SEED-v004-val --model g1 --mode hybrid --checkpoint "$checkpoint" \
    --manifest "$MANIFEST" --stats-dir "$TRAIN" --text-cache "$TEXT" --protocol "$protocol" \
    --total-seconds 9 --selection-min-seconds 9 --samples 64 --batch-size 2 \
    --diffusion-steps 10 --cfg-weight-text 2 --cfg-weight-constraint 2 --seed "$SEED" \
    --device cuda:0 --output "$OUT/$stem.json" --artifacts-output "$OUT/$stem.npz" \
    >"$OUT/$stem.log" 2>&1
}

run_long() {
  local gpu=$1 mode=$2 seconds=$3
  local stem=${mode}_${seconds}s_in_horizon
  local checkpoint=$CURRENT/$mode/checkpoint_step_000900.pt diag=()
  if result_pass "$OUT/$stem.json" 16 && [[ -f $OUT/$stem.npz ]]; then
    if [[ $mode != hybrid || $seconds != 40 || -f $OUT/$stem.state.jsonl ]]; then return; fi
  fi
  if [[ $mode == hybrid && $seconds == 40 ]]; then
    diag=(--state-diagnostics-output "$OUT/$stem.state.jsonl")
  fi
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$ROOT/examples/evaluate_ardy_official_table5.py" \
    --dataset-name BONES-SEED-v004-val --model g1 --mode "$mode" --checkpoint "$checkpoint" \
    --manifest "$MANIFEST" --stats-dir "$TRAIN" --text-cache "$TEXT" --protocol in_horizon \
    --total-seconds "$seconds" --selection-min-seconds 40 --samples 16 --batch-size 2 \
    --diffusion-steps 10 --cfg-weight-text 2 --cfg-weight-constraint 2 --seed "$SEED" \
    --device cuda:0 --output "$OUT/$stem.json" --artifacts-output "$OUT/$stem.npz" \
    "${diag[@]}" >"$OUT/$stem.log" 2>&1
}

pids=()
run_table4 0 healthy "$HEALTHY_HYBRID" & pids+=("$!")
run_9s 1 healthy "$HEALTHY_HYBRID" in_horizon & pids+=("$!")
run_9s 2 healthy "$HEALTHY_HYBRID" out_of_horizon & pids+=("$!")
run_table4 3 stage900 "$CURRENT/hybrid/checkpoint_step_000900.pt" & pids+=("$!")
run_9s 4 stage900 "$CURRENT/hybrid/checkpoint_step_000900.pt" in_horizon & pids+=("$!")
run_9s 5 stage900 "$CURRENT/hybrid/checkpoint_step_000900.pt" out_of_horizon & pids+=("$!")
(run_long 6 minimal 20; run_long 6 minimal 40) & pids+=("$!")
(run_long 7 hybrid 20; run_long 7 hybrid 40) & pids+=("$!")
for pid in "${pids[@]}"; do wait "$pid"; done
date -Is >"$OUT/EVALUATION_COMPLETE"

"$PY" - "$OUT" "$REPORT" <<'PY'
import json, sys
from pathlib import Path
from examples.check_bones_seed_generated_history_gate import (
    paired_bootstrap_gap, sample_root_slopes, state_audit,
)

out, report = map(Path, sys.argv[1:])
metrics = ("constraint_error_cm_mean", "mobench_root_ade_cm_mean")
retention = {}
retention_pass = True
for stem in ("table4", "9s_in_horizon", "9s_out_of_horizon"):
    base = json.loads((out / f"healthy_{stem}.json").read_text())
    current = json.loads((out / f"stage900_{stem}.json").read_text())
    retention[stem] = {}
    for metric in metrics:
        change = current[metric] / max(base[metric], 1e-8) - 1
        passed = change <= 0.10
        retention[stem][metric] = {
            "baseline": base[metric], "current": current[metric],
            "relative_change": change, "passed": passed,
        }
        retention_pass &= passed

long = {}
long_pass = True
for seconds in (20, 40):
    hybrid = json.loads((out / f"hybrid_{seconds}s_in_horizon.json").read_text())
    minimal = json.loads((out / f"minimal_{seconds}s_in_horizon.json").read_text())
    gaps = {
        "constraint_mean_gap_cm": hybrid["constraint_error_cm_mean"] - minimal["constraint_error_cm_mean"],
        "root_ade_gap_cm": hybrid["mobench_root_ade_cm_mean"] - minimal["mobench_root_ade_cm_mean"],
        "root_slope_gap_cm_per_window": hybrid["post_window5_root_ade_slope_cm_per_window"] - minimal["post_window5_root_ade_slope_cm_per_window"],
        "constraint_slope_gap_cm_per_window": hybrid["post_window5_constraint_slope_cm_per_window"] - minimal["post_window5_constraint_slope_cm_per_window"],
    }
    paired = paired_bootstrap_gap(
        sample_root_slopes(out / f"hybrid_{seconds}s_in_horizon.npz", int(hybrid["generation_horizon_frames"])),
        sample_root_slopes(out / f"minimal_{seconds}s_in_horizon.npz", int(minimal["generation_horizon_frames"])),
        seed=20260829 + seconds,
    )
    passed = max(gaps.values()) < 0
    long_pass &= passed
    long[f"{seconds}s"] = {"gaps": gaps, "paired_root_slope_gap": paired, "passed": passed}

state = state_audit(out / "hybrid_40s_in_horizon.state.jsonl")
passed = retention_pass and long_pass and state["all_finite"] and state["no_explosion"]
gate = {
    "status": "PASS" if passed else "FAIL",
    "split": "validation",
    "seed": 20260829,
    "long_samples": 16,
    "criteria": {
        "short_retention": retention_pass,
        "hybrid_20_40_means_and_slopes_better": long_pass,
        "state_finite": state["all_finite"],
        "no_state_norm_explosion": state["no_explosion"],
    },
    "short_retention": retention,
    "long_horizon": long,
    "state_audit": state,
}
(out / "gate.json").write_text(json.dumps(gate, indent=2) + "\n")

lines = [
    "# BONES-SEED Step-900 Validation Confirmation",
    "",
    "This confirmation uses the actor-disjoint validation split, seed 20260829, all 16 clips with at least 1000 frames for 20/40-second evaluation, and 64 samples for Table 4/9-second retention. It was not used by the checkpoint sweep, but the checkpoints' informational dev loss was previously computed on this split.",
    "",
    f"**{gate['status']}**",
    "",
    "| Criterion | Passed |",
    "|---|---|",
]
for key, value in gate["criteria"].items():
    lines.append(f"| {key.replace('_', ' ')} | {value} |")
lines.extend(["", "| Duration | Constraint mean gap | Root ADE gap | Root slope gap | Constraint slope gap | Paired Root-slope 95% CI |", "|---:|---:|---:|---:|---:|---:|"])
for seconds in (20, 40):
    row = long[f"{seconds}s"]
    gaps, paired = row["gaps"], row["paired_root_slope_gap"]
    lines.append(
        f"| {seconds} s | {gaps['constraint_mean_gap_cm']:.2f} | {gaps['root_ade_gap_cm']:.2f} | "
        f"{gaps['root_slope_gap_cm_per_window']:.3f} | {gaps['constraint_slope_gap_cm_per_window']:.3f} | "
        f"[{paired['ci95_cm_per_window'][0]:.3f}, {paired['ci95_cm_per_window'][1]:.3f}] |"
    )
lines.extend(["", "Negative gaps favor Hybrid.", ""])
report.parent.mkdir(parents=True, exist_ok=True)
report.write_text("\n".join(lines))
print(json.dumps(gate, indent=2))
print(report)
PY

date -Is >"$OUT/CONFIRMATION_COMPLETE"
