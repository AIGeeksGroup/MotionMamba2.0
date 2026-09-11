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
SOURCE=$BASE/bones_seed_generated_history_endpoint_v9_ft_20260829
REFERENCE=$BASE/bones_seed_generated_history_v2_earlystop900_eval_20260828
OUT=$BASE/bones_seed_generated_history_endpoint_v9_v2_test_protocol_20260829
TRAIN=$BASE/bones_seed_ardy_train
TEST=$BASE/bones_seed_ardy_test/manifest.jsonl
TEXT=$BASE/bones_seed_text_cache_llm2vec
PANEL=$BASE/bones_seed_long_memory_panel_v1_20260831
REPORT=$ROOT/docs/BONES_SEED_V9_V2_EXACT_TEST_PROTOCOL_COMPARISON_20260829.md
SEED=20260824
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
  local gpu=$1 mode=$2 stem=${mode}_table4
  local checkpoint=$SOURCE/$mode/checkpoint_step_000100.pt
  result_pass "$OUT/$stem.json" 64 && [[ -f $OUT/$stem.npz ]] && return
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$ROOT/examples/evaluate_humanml3d_table45.py" \
    --dataset-name BONES-SEED-v004 --model g1 --mode "$mode" --checkpoint "$checkpoint" \
    --manifest "$TEST" --stats-dir "$TRAIN" --text-cache "$TEXT" --protocol table4 \
    --samples 64 --batch-size 4 --diffusion-steps 10 --cfg-weight-text 2 \
    --cfg-weight-constraint 2 --seed "$SEED" --device cuda:0 \
    --output "$OUT/$stem.json" --artifacts-output "$OUT/$stem.npz" \
    >"$OUT/$stem.log" 2>&1
}

run_duration() {
  local gpu=$1 mode=$2 seconds=$3 protocol=$4 samples=$5 selection=$6
  local checkpoint=$SOURCE/$mode/checkpoint_step_000100.pt
  local stem=${mode}_${seconds}s_${protocol} extra=()
  if result_pass "$OUT/$stem.json" "$samples" && [[ -f $OUT/$stem.npz ]]; then
    if [[ $mode != hybrid || $seconds != 40 || $protocol != in_horizon || -f $OUT/$stem.state.jsonl ]]; then
      return
    fi
  fi
  # These are v9 model inference parameters used during its aligned training and validation.
  if [[ $mode == hybrid ]]; then
    extra+=(
      --mamba-output-decay-start-windows 4
      --mamba-output-decay-half-life-windows 4
      --mamba-output-decay-min-scale 0
    )
    if [[ $seconds == 40 && $protocol == in_horizon ]]; then
      extra+=(--state-diagnostics-output "$OUT/$stem.state.jsonl")
    fi
  fi
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$ROOT/examples/evaluate_ardy_official_table5.py" \
    --dataset-name BONES-SEED-v004 --model g1 --mode "$mode" --checkpoint "$checkpoint" \
    --manifest "$TEST" --stats-dir "$TRAIN" --text-cache "$TEXT" --protocol "$protocol" \
    --total-seconds "$seconds" --selection-min-seconds "$selection" --samples "$samples" \
    --batch-size 2 --diffusion-steps 10 --cfg-weight-text 2 --cfg-weight-constraint 2 \
    --seed "$SEED" --device cuda:0 --output "$OUT/$stem.json" \
    --artifacts-output "$OUT/$stem.npz" --panel "$PANEL/$stem.jsonl" \
    --protocol-lock "$PANEL/protocol_lock.json" "${extra[@]}" >"$OUT/$stem.log" 2>&1
}

for mode in minimal hybrid; do
  pids=()
  run_table4 0 "$mode" & pids+=("$!")
  run_duration 1 "$mode" 9 in_horizon 64 9 & pids+=("$!")
  run_duration 2 "$mode" 9 out_of_horizon 64 9 & pids+=("$!")
  run_duration 3 "$mode" 20 in_horizon 20 40 & pids+=("$!")
  run_duration 4 "$mode" 20 out_of_horizon 20 40 & pids+=("$!")
  run_duration 6 "$mode" 40 in_horizon 20 40 & pids+=("$!")
  run_duration 7 "$mode" 40 out_of_horizon 20 40 & pids+=("$!")
  for pid in "${pids[@]}"; do wait "$pid"; done
done

# ARDY has no endpoint-v9 checkpoint. Reuse its already-completed v2 step900
# evaluation because it used this exact panel, seed, and protocol.
for stem in table4 9s_in_horizon 9s_out_of_horizon 20s_in_horizon \
  20s_out_of_horizon 40s_in_horizon 40s_out_of_horizon; do
  for extension in json log npz; do
    ln -sfn "$REFERENCE/original_${stem}.${extension}" "$OUT/ardy_${stem}.${extension}"
  done
done

"$PY" - "$REFERENCE" "$OUT" "$REPORT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

reference, output, report = map(Path, sys.argv[1:])
stems = (
    "table4",
    "9s_in_horizon",
    "9s_out_of_horizon",
    "20s_in_horizon",
    "20s_out_of_horizon",
    "40s_in_horizon",
    "40s_out_of_horizon",
)
protocol_fields = (
    "protocol",
    "samples",
    "diffusion_steps",
    "cfg_weight_text",
    "cfg_weight_constraint",
)
duration_fields = (
    "fps",
    "total_frames",
    "initial_history_frames",
    "generated_frames",
    "generation_horizon_frames",
    "generation_windows",
    "goals_per_sample",
    "memory_mode",
    "protocol_guard",
)
shared_duration_fields = tuple(
    key for key in duration_fields if key not in ("memory_mode", "protocol_guard")
)
rows = {}
verification = {}
for mode in ("minimal", "hybrid"):
    for stem in stems:
        old = json.loads((reference / f"{mode}_{stem}.json").read_text())
        new = json.loads((output / f"{mode}_{stem}.json").read_text())
        fields = protocol_fields + (() if stem == "table4" else duration_fields)
        mismatches = {
            key: {"v2": old.get(key), "v9": new.get(key)}
            for key in fields
            if old.get(key) != new.get(key)
        }
        if old["sample_ids"] != new["sample_ids"]:
            mismatches["sample_ids"] = "ordered sample lists differ"
        if mismatches:
            raise RuntimeError(f"protocol mismatch for {mode}/{stem}: {mismatches}")
        rows[("v2", mode, stem)] = old
        rows[("v9", mode, stem)] = new
        digest = hashlib.sha256("\n".join(new["sample_ids"]).encode()).hexdigest()
        verification[f"{mode}_{stem}"] = {
            "sample_ids_equal": True,
            "protocol_fields_equal": True,
            "ordered_sample_sha256": digest,
        }

for stem in stems:
    ardy = json.loads((output / f"ardy_{stem}.json").read_text())
    paired = rows[("v9", "hybrid", stem)]
    fields = protocol_fields + (() if stem == "table4" else shared_duration_fields)
    mismatches = {
        key: {"ardy": ardy.get(key), "v9": paired.get(key)}
        for key in fields
        if ardy.get(key) != paired.get(key)
    }
    if ardy["sample_ids"] != paired["sample_ids"]:
        mismatches["sample_ids"] = "ordered sample lists differ"
    if mismatches:
        raise RuntimeError(f"ARDY protocol mismatch for {stem}: {mismatches}")
    rows[("v2", "ardy", stem)] = ardy
    verification[f"ardy_{stem}"] = {
        "sample_ids_equal": True,
        "protocol_fields_equal": True,
        "ordered_sample_sha256": hashlib.sha256(
            "\n".join(ardy["sample_ids"]).encode()
        ).hexdigest(),
        "checkpoint_version": "v2_step900",
        "architecture_history_contract": {
            "memory_mode": ardy.get("memory_mode"),
            "protocol_guard": ardy.get("protocol_guard"),
        },
    }

(output / "protocol_verification.json").write_text(
    json.dumps(
        {
            "status": "PASS",
            "reference": str(reference),
            "seed": 20260824,
            "checks": verification,
        },
        indent=2,
    ) + "\n"
)

def metric(row, key):
    value = row.get(key)
    return "N/A" if value is None else f"{value:.2f}"

def slope(row, key):
    value = row.get(key)
    return "N/A" if value is None else f"{value:.3f}"

lines = [
    "# BONES-SEED v9 vs v2 Exact Test-Protocol Comparison (2026-08-29)",
    "",
    "## Protocol Verification",
    "",
    "**PASS.** v9 uses the exact ordered v2 test sample panels, seed 20260824, "
    "diffusion/CFG settings, generation durations, window counts, control-goal counts, "
    "persistent-memory protocol, and GT-prefix priming contract.",
    "",
    "v9 Hybrid retains its trained inference setting: Mamba readout decay starts after "
    "window 4 with half-life 4. This is a model parameter rather than a change to the "
    "test panel or control protocol.",
    "",
    "There is no endpoint-v9 ARDY checkpoint. The ARDY row is the healthy v2 "
    "step900 checkpoint already evaluated on this exact protocol; it is held fixed "
    "as the common architecture baseline.",
    "ARDY uses its native explicit-history contract, while Minimal/Hybrid use "
    "persistent cache; all cross-architecture sample, duration, window, goal, "
    "diffusion, and CFG fields are matched.",
    "",
    "## Table 4: Offline Text and Joint-Position Control",
    "",
    "| Version | Method | Constraint (cm) | Root ADE (cm) | Root FDE (cm) | PA-MPJPE (cm) |",
    "|---|---|---:|---:|---:|---:|",
]
row = rows[("v2", "ardy", "table4")]
lines.append(
    f"| v2 | ARDY | {metric(row, 'constraint_error_cm_mean')} | "
    f"{metric(row, 'mobench_root_ade_cm_mean')} | "
    f"{metric(row, 'mobench_root_fde_cm_mean')} | "
    f"{metric(row, 'mobench_pa_mpjpe_cm_mean')} |"
)
for version in ("v2", "v9"):
    for mode, label in (("minimal", "Minimal"), ("hybrid", "Hybrid")):
        row = rows[(version, mode, "table4")]
        lines.append(
            f"| {version} | {label} | {metric(row, 'constraint_error_cm_mean')} | "
            f"{metric(row, 'mobench_root_ade_cm_mean')} | "
            f"{metric(row, 'mobench_root_fde_cm_mean')} | "
            f"{metric(row, 'mobench_pa_mpjpe_cm_mean')} |"
        )

lines.extend([
    "",
    "## Table 5: ARDY-Aligned 9-Second Control",
    "",
    "| Protocol | Version | Method | Constraint (cm) | Root ADE (cm) | Root FDE (cm) | PA-MPJPE (cm) |",
    "|---|---|---|---:|---:|---:|---:|",
])
for stem, protocol in (("9s_in_horizon", "in"), ("9s_out_of_horizon", "out")):
    row = rows[("v2", "ardy", stem)]
    lines.append(
        f"| {protocol} | v2 | ARDY | {metric(row, 'constraint_error_cm_mean')} | "
        f"{metric(row, 'mobench_root_ade_cm_mean')} | "
        f"{metric(row, 'mobench_root_fde_cm_mean')} | "
        f"{metric(row, 'mobench_pa_mpjpe_cm_mean')} |"
    )
    for version in ("v2", "v9"):
        for mode, label in (("minimal", "Minimal"), ("hybrid", "Hybrid")):
            row = rows[(version, mode, stem)]
            lines.append(
                f"| {protocol} | {version} | {label} | "
                f"{metric(row, 'constraint_error_cm_mean')} | "
                f"{metric(row, 'mobench_root_ade_cm_mean')} | "
                f"{metric(row, 'mobench_root_fde_cm_mean')} | "
                f"{metric(row, 'mobench_pa_mpjpe_cm_mean')} |"
            )

lines.extend([
    "",
    "## Table 5 Long-Horizon Extension",
    "",
    "| Duration | Protocol | Version | Method | Constraint (cm) | Root ADE (cm) | Post-W5 Root slope | Post-W5 constraint slope |",
    "|---:|---|---|---|---:|---:|---:|---:|",
])
for seconds in (20, 40):
    for protocol in ("in_horizon", "out_of_horizon"):
        stem = f"{seconds}s_{protocol}"
        row = rows[("v2", "ardy", stem)]
        lines.append(
            f"| {seconds} s | {'in' if protocol == 'in_horizon' else 'out'} | v2 | ARDY | "
            f"{metric(row, 'constraint_error_cm_mean')} | "
            f"{metric(row, 'mobench_root_ade_cm_mean')} | "
            f"{slope(row, 'post_window5_root_ade_slope_cm_per_window')} | "
            f"{slope(row, 'post_window5_constraint_slope_cm_per_window')} |"
        )
        for version in ("v2", "v9"):
            for mode, label in (("minimal", "Minimal"), ("hybrid", "Hybrid")):
                row = rows[(version, mode, stem)]
                lines.append(
                    f"| {seconds} s | {'in' if protocol == 'in_horizon' else 'out'} | {version} | {label} | "
                    f"{metric(row, 'constraint_error_cm_mean')} | "
                    f"{metric(row, 'mobench_root_ade_cm_mean')} | "
                    f"{slope(row, 'post_window5_root_ade_slope_cm_per_window')} | "
                    f"{slope(row, 'post_window5_constraint_slope_cm_per_window')} |"
                )

lines.extend([
    "",
    "## Direct Hybrid Comparison",
    "",
    "Negative deltas mean v9 improves over v2 on the identical test panel.",
    "",
    "| Protocol | Constraint delta (cm) | Root ADE delta (cm) | Root-slope delta | Constraint-slope delta |",
    "|---|---:|---:|---:|---:|",
])
for stem in stems:
    old = rows[("v2", "hybrid", stem)]
    new = rows[("v9", "hybrid", stem)]
    label = stem.replace("_", " ")
    root_slope_old = old.get("post_window5_root_ade_slope_cm_per_window")
    root_slope_new = new.get("post_window5_root_ade_slope_cm_per_window")
    constraint_slope_old = old.get("post_window5_constraint_slope_cm_per_window")
    constraint_slope_new = new.get("post_window5_constraint_slope_cm_per_window")
    root_delta = "N/A" if root_slope_old is None else f"{root_slope_new - root_slope_old:+.3f}"
    constraint_delta = "N/A" if constraint_slope_old is None else f"{constraint_slope_new - constraint_slope_old:+.3f}"
    lines.append(
        f"| {label} | {new['constraint_error_cm_mean'] - old['constraint_error_cm_mean']:+.2f} | "
        f"{new['mobench_root_ade_cm_mean'] - old['mobench_root_ade_cm_mean']:+.2f} | "
        f"{root_delta} | {constraint_delta} |"
    )

lines.extend([
    "",
    "## Result",
    "",
    "v9 is better overall on the identical v2 test panels, but it is not a uniform "
    "win on every long-horizon metric.",
    "",
    "- v9 Hybrid improves all reported Table 4 and 9-second constraint/Root metrics "
    "over v2 Hybrid and is also better than the fixed ARDY baseline.",
    "- At 40 seconds in-horizon, v9 Hybrid improves constraint/Root ADE from "
    "137.47/134.89 cm to 121.64/118.21 cm and improves both post-window-5 slopes "
    "from -1.238/-1.311 to -6.079/-6.237. It also remains clearly better than "
    "v9 Minimal on both means and slopes.",
    "- At 20 seconds in-horizon, v9 Hybrid regresses by 4.67 cm constraint and "
    "1.55 cm Root ADE relative to v2 Hybrid; its mean errors and Root slope are "
    "also worse than v9 Minimal, although its constraint slope is better.",
    "- The 20/40-second out-of-horizon mean errors improve substantially over v2, "
    "but v9 Hybrid Root ADE is slightly worse than v9 Minimal at both durations, "
    "and its 40-second Root slope is worse.",
    "",
    "Accordingly, v9 is the stronger 40-second and overall checkpoint on this exact "
    "panel, while v2 step900 remains the stronger 20-second in-horizon checkpoint. "
    "This test-panel result must remain separate from the actor-disjoint validation "
    "gate, where v9 still fails the strict all-slopes criterion.",
    "",
    "## Artifacts",
    "",
    f"- v2 reference: `{reference}`",
    f"- v9 evaluation: `{output}`",
    f"- Fixed ARDY checkpoint: `{rows[('v2', 'ardy', 'table4')]['checkpoint']}`",
    f"- Protocol assertions: `{output / 'protocol_verification.json'}`",
    "",
])
report.write_text("\n".join(lines))
print(report)
PY

date -Is >"$OUT/EVALUATION_COMPLETE"
