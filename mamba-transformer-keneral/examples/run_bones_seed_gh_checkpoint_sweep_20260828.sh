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
OUT=$BASE/bones_seed_gh_checkpoint_sweep_20260828
TRAIN=$BASE/bones_seed_ardy_train
TEST=$BASE/bones_seed_ardy_test/manifest.jsonl
TEXT=$BASE/bones_seed_text_cache_llm2vec

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

run_task() {
  local gpu=$1 run=$2 stage=$3 mode=$4 task=$5
  local checkpoint=$BASE/$run/$mode/checkpoint_step_$(printf '%06d' "$stage").pt
  local out=$OUT/$run/stage$stage stem samples seconds
  mkdir -p "$out"
  case "$task" in
    table4)
      stem=${mode}_table4
      result_pass "$out/$stem.json" 64 && return
      CUDA_VISIBLE_DEVICES=$gpu "$PY" "$ROOT/examples/evaluate_humanml3d_table45.py" \
        --dataset-name BONES-SEED-v004 --model g1 --mode "$mode" --checkpoint "$checkpoint" \
        --manifest "$TEST" --stats-dir "$TRAIN" --text-cache "$TEXT" --protocol table4 \
        --samples 64 --batch-size 4 --diffusion-steps 10 --cfg-weight-text 2 \
        --cfg-weight-constraint 2 --seed 20260824 --device cuda:0 \
        --output "$out/$stem.json" >"$out/$stem.log" 2>&1
      ;;
    9in|9out|20in|40in)
      seconds=${task%%in}; seconds=${seconds%%out}
      [[ $task == *out ]] && protocol=out_of_horizon || protocol=in_horizon
      samples=20
      [[ $seconds == 9 ]] && samples=64
      stem=${mode}_${seconds}s_${protocol}
      result_pass "$out/$stem.json" "$samples" && return
      CUDA_VISIBLE_DEVICES=$gpu "$PY" "$ROOT/examples/evaluate_ardy_official_table5.py" \
        --dataset-name BONES-SEED-v004 --model g1 --mode "$mode" --checkpoint "$checkpoint" \
        --manifest "$TEST" --stats-dir "$TRAIN" --text-cache "$TEXT" --protocol "$protocol" \
        --total-seconds "$seconds" --selection-min-seconds "$([[ $seconds == 9 ]] && echo 9 || echo 40)" \
        --samples "$samples" --batch-size 2 --diffusion-steps 10 --cfg-weight-text 2 \
        --cfg-weight-constraint 2 --seed 20260824 --device cuda:0 \
        --output "$out/$stem.json" >"$out/$stem.log" 2>&1
      ;;
  esac
}

tasks=()
for run in bones_seed_generated_history_v2_ft_20260828 bones_seed_generated_history_v3_ft_20260828; do
  for stage in 600 700 800 900; do
    tasks+=("$run $stage minimal 20in")
    tasks+=("$run $stage minimal 40in")
    tasks+=("$run $stage hybrid table4")
    tasks+=("$run $stage hybrid 9in")
    tasks+=("$run $stage hybrid 9out")
    tasks+=("$run $stage hybrid 20in")
    tasks+=("$run $stage hybrid 40in")
  done
done

pids=()
for gpu in 0 1 2 3 4 5 6 7; do
  (
    for ((index=gpu; index<${#tasks[@]}; index+=8)); do
      read -r run stage mode task <<<"${tasks[$index]}"
      run_task "$gpu" "$run" "$stage" "$mode" "$task"
    done
  ) & pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

"$PY" - "$OUT" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
baseline_root = Path("/data4/zts/result/bones_seed_ardy_retrained_official_table5_20260826")
baseline_table4 = json.loads(Path(
    "/data4/zts/result/bones_seed_ardy_retrained_dual_eval_20260826/hybrid_table4.json"
).read_text())
baselines = {
    "table4": baseline_table4,
    "9s_in_horizon": json.loads((baseline_root / "hybrid_9s_in_horizon.json").read_text()),
    "9s_out_of_horizon": json.loads((baseline_root / "hybrid_9s_out_of_horizon.json").read_text()),
}
metrics = ("constraint_error_cm_mean", "mobench_root_ade_cm_mean")
rows = []
for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
    for stage_dir in sorted(run_dir.glob("stage*")):
        retention = max(
            json.loads((stage_dir / f"hybrid_{stem}.json").read_text())[metric]
            / max(baseline[metric], 1e-8) - 1
            for stem, baseline in baselines.items()
            for metric in metrics
        )
        long = {}
        passed = retention <= 0.10
        for seconds in (20, 40):
            hybrid = json.loads((stage_dir / f"hybrid_{seconds}s_in_horizon.json").read_text())
            minimal = json.loads((stage_dir / f"minimal_{seconds}s_in_horizon.json").read_text())
            root_gap = hybrid["post_window5_root_ade_slope_cm_per_window"] - minimal["post_window5_root_ade_slope_cm_per_window"]
            constraint_gap = hybrid["post_window5_constraint_slope_cm_per_window"] - minimal["post_window5_constraint_slope_cm_per_window"]
            ade_gap = hybrid["mobench_root_ade_cm_mean"] - minimal["mobench_root_ade_cm_mean"]
            constraint_mean_gap = hybrid["constraint_error_cm_mean"] - minimal["constraint_error_cm_mean"]
            long[seconds] = {
                "root_slope_gap": root_gap,
                "constraint_slope_gap": constraint_gap,
                "root_ade_gap": ade_gap,
                "constraint_mean_gap": constraint_mean_gap,
            }
            passed &= max(root_gap, constraint_gap, ade_gap, constraint_mean_gap) < 0
        rows.append({
            "run": run_dir.name,
            "stage": int(stage_dir.name.removeprefix("stage")),
            "max_short_degradation": retention,
            "long_horizon": long,
            "all_mean_and_slope_checks_pass": passed,
        })
(root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
print(json.dumps(rows, indent=2))
PY

date -Is >"$OUT/SWEEP_COMPLETE"
