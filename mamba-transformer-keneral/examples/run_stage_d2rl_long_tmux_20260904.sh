#!/usr/bin/env bash
set -euo pipefail

ROOT=/data3/zts/MotionMamba2.0/mamba-transformer-keneral
PY=/data3/zts/conda_envs/ardy_env/bin/python
BASE=/data4/zts/result
RUN_ROOT=$BASE/stage_d2rl_bones_long_20260904
PANEL_ROOT=$BASE/bones_seed_long_memory_panel_v2_20260831/selection
PROTOCOL_LOCK=$ROOT/configs/bones_seed_long_memory_protocol_v2.json
TRAIN_MANIFEST=$BASE/bones_seed_ardy_train/manifest.jsonl
VAL_MANIFEST=$BASE/bones_seed_ardy_val/manifest.jsonl
STATS=$BASE/bones_seed_ardy_train
TEXT=$BASE/bones_seed_text_cache_llm2vec
NATIVE_HEALTH=$BASE/bones_seed_long_memory_stage_b_selection_eval_20260831/stage000/hybrid_no_decay_carry

FULL_RESUME=$BASE/stage_d2r_bones_step250_full_20260903/checkpoint_step_000250.pt
NO_DELTA_RESUME=$BASE/stage_d2r_bones_step250_no_delta_20260903/checkpoint_step_000250.pt
TARGETS=(500 1000 2000 5000)

export HF_HOME=/data3/zts/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LOCAL_CACHE=true
export PYTHONPATH=$ROOT/src:/data3/zts/MotionMamba2.0/ardy:/data3/zts/packages/state-spaces-mamba:/data3/zts/packages/mamba_op_triton${PYTHONPATH:+:$PYTHONPATH}

[[ ! -e $RUN_ROOT ]] || {
  echo "refusing to overwrite existing run root: $RUN_ROOT" >&2
  exit 1
}
mkdir -p "$RUN_ROOT"

"$PY" - "$RUN_ROOT" "$ROOT" "$FULL_RESUME" "$NO_DELTA_RESUME" <<'PY'
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

out, root, full, no_delta = map(Path, sys.argv[1:])

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

record = {
    "schema_version": "d2rl_long_training_v1",
    "started_at": datetime.now().astimezone().isoformat(),
    "git_sha": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip(),
    "git_worktree_dirty": bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True
        ).strip()
    ),
    "resume_checkpoints": {"full": str(full), "no_delta": str(no_delta)},
    "resume_sha256": {"full": sha256(full), "no_delta": sha256(no_delta)},
    "launcher": str(root / "examples/run_stage_d2rl_long_tmux_20260904.sh"),
    "launcher_sha256": sha256(root / "examples/run_stage_d2rl_long_tmux_20260904.sh"),
    "targets_total_optimizer_steps": [500, 1000, 2000, 5000],
    "locked_final_actors_read": False,
    "experiment_scope": "exploratory matched long-training continuation",
}
(out / "experiment_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
PY

COMMON=(
  --model g1 --mode hybrid --manifest "$TRAIN_MANIFEST" --dev-manifest "$VAL_MANIFEST"
  --stats-dir "$STATS" --text-cache "$TEXT" --batch-size 1 --learning-rate 1e-4
  --seed 9032026 --frames 312 --history-probability 0 --constraint-dropout 0.1
  --text-dropout 0.1 --sequential-cache-probability 1
  --sequential-rollout-min-windows 6 --sequential-rollout-windows 6
  --sequential-rollout-curriculum d2_post_eviction_v1
  --generated-history-probability-start 1 --generated-history-probability-1k 1
  --generated-history-probability-5k 1 --generated-history-diffusion-steps 10
  --generated-history-max-supervision-points 1 --generated-history-bptt-windows 4
  --generated-history-attach-seed-write --eval-protocol explicit_history
  --eval-batches 1 --eval-every 0 --keep-last-checkpoints 1
  --no-final-resume-checkpoint --boundary-loss-weight 10
  --root-velocity-loss-weight 50 --constraint-loss-weight 1
  --fk-constraint-loss-weight 1 --foot-skate-loss-weight 50
  --train-scope state_memory --attention-window-chunks 4
  --token-state-interaction state_query --state-aware-output-mode native_gated_add
  --state-summary-dim 128 --use-state-delta --state-conditioning-scale-init 0
  --content-memory-enabled --content-memory-slots 2
  --content-memory-write-gate-context previous_memory
  --content-memory-decoder-init-scale 0.02
  --content-memory-write-gate-init-scale 0.02
  --content-memory-write-gate-bias-init 0 --content-memory-read-scale-init 0
  --content-memory-conditioning text --content-memory-normalize-delta
  --content-memory-normalize-slots --content-memory-max-write-rate 0.25
  --log-every 10
)

run_pair_stage() {
  local target=$1
  local stage=$RUN_ROOT/step${target}
  local full_out=$stage/full
  local no_delta_out=$stage/no_delta
  mkdir -p "$stage"
  [[ ! -e $full_out && ! -e $no_delta_out ]] || {
    echo "refusing to reuse stage outputs: $stage" >&2
    return 1
  }

  CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run --standalone \
    --nproc-per-node=4 "$ROOT/examples/train_humanml3d_ardy_ddp.py" \
    "${COMMON[@]}" --steps "$target" --save-every "$target" \
    --content-memory-use-state-delta --resume "$FULL_RESUME" \
    --output-dir "$full_out" >"$stage/full_train.log" 2>&1 &
  local full_pid=$!
  CUDA_VISIBLE_DEVICES=4,5,6,7 "$PY" -m torch.distributed.run --standalone \
    --nproc-per-node=4 "$ROOT/examples/train_humanml3d_ardy_ddp.py" \
    "${COMMON[@]}" --steps "$target" --save-every "$target" \
    --no-content-memory-use-state-delta --resume "$NO_DELTA_RESUME" \
    --output-dir "$no_delta_out" >"$stage/no_delta_train.log" 2>&1 &
  local no_delta_pid=$!

  local status=0
  wait "$full_pid" || status=1
  wait "$no_delta_pid" || status=1
  (( status == 0 )) || return "$status"

  "$PY" - "$target" "$full_out/metrics.json" "$no_delta_out/metrics.json" \
    "$stage/training_safety_gate.json" <<'PY'
import json
import math
import sys
from pathlib import Path

target = int(sys.argv[1])
paths = {"full": Path(sys.argv[2]), "no_delta": Path(sys.argv[3])}
output = Path(sys.argv[4])
checks = {}
summaries = {}
for name, path in paths.items():
    data = json.loads(path.read_text())
    diag = data["d2_last_memory_diagnostics"]
    fractions = data["d2_gradient_update_fraction"]
    finite_diag = all(
        value is None or math.isfinite(float(value)) for value in diag.values()
    )
    variant_checks = {
        "reached_target_step": int(data["steps"]) == target,
        "all_gradients_finite": bool(data["all_gradients_finite"]),
        "diagnostics_finite": finite_diag,
        "decoder_write_read_update_fraction_ge_0_90": all(
            float(value) >= 0.90 for value in fractions.values()
        ),
        "write_gate_within_architectural_cap": (
            0.0 <= float(diag["write_gate_min"])
            and float(diag["write_gate_max"]) <= 0.250001
        ),
        "memory_slot_norm_le_40": float(diag["memory_slot_norm"]) <= 40.0,
        "content_proposal_norm_le_40": float(diag["content_proposal_norm"]) <= 40.0,
        "read_residual_scale_abs_le_0_10": abs(float(diag["read_residual_scale"])) <= 0.10,
    }
    checks[name] = variant_checks
    summaries[name] = {
        "trainer_status_informational": data["status"],
        "first_50_mean_loss": data["first_50_mean_loss"],
        "last_50_mean_loss": data["last_50_mean_loss"],
        "final_eval_loss": data["final_eval_loss"],
        "gradient_update_fraction": fractions,
        "memory_diagnostics": diag,
    }
passed = all(all(group.values()) for group in checks.values())
result = {
    "status": "PASS" if passed else "FAIL",
    "target_total_optimizer_steps": target,
    "checks": checks,
    "summaries": summaries,
    "note": "trainer_status is retained but random-minibatch trend is not the hard safety gate",
}
output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
raise SystemExit(0 if passed else 1)
PY

  run_health_eval "$target" full "$full_out/hybrid_humanml3d_ddp.pt" 0 1
  local full_health_pid=$!
  run_health_eval "$target" no_delta "$no_delta_out/hybrid_humanml3d_ddp.pt" 2 3
  local no_delta_health_pid=$!
  status=0
  wait "$full_health_pid" || status=1
  wait "$no_delta_health_pid" || status=1
  (( status == 0 )) || return "$status"
  validate_health_gate "$target" "$stage"

  FULL_RESUME=$full_out/checkpoint_step_$(printf '%06d' "$target").pt
  NO_DELTA_RESUME=$no_delta_out/checkpoint_step_$(printf '%06d' "$target").pt
  date -Is >"$stage/STAGE_COMPLETE"
}

run_health_eval() {
  local target=$1 variant=$2 checkpoint=$3 table_gpu=$4 long_gpu=$5
  local out=$RUN_ROOT/step${target}/health/$variant
  mkdir -p "$out/table4" "$out/9s_in_horizon"
  (
    CUDA_VISIBLE_DEVICES=$table_gpu "$PY" "$ROOT/examples/evaluate_humanml3d_table45.py" \
      --mode hybrid --model g1 --dataset-name BONES-SEED-v004 \
      --checkpoint "$checkpoint" --manifest "$VAL_MANIFEST" --stats-dir "$STATS" \
      --text-cache "$TEXT" --protocol table4 --samples 64 --batch-size 2 \
      --diffusion-steps 10 --cfg-weight-text 2 --cfg-weight-constraint 2 \
      --seed 20260831 --device cuda:0 --panel "$PANEL_ROOT/table4.jsonl" \
      --protocol-lock "$PROTOCOL_LOCK" --attention-window-chunks 4 \
      --output "$out/table4/result.json" \
      --per-sample-output "$out/table4/per_sample.jsonl" \
      >"$out/table4/eval.log" 2>&1
  ) &
  local table_pid=$!
  (
    CUDA_VISIBLE_DEVICES=$long_gpu "$PY" "$ROOT/examples/evaluate_ardy_official_table5.py" \
      --mode hybrid --model g1 --dataset-name BONES-SEED-v004 \
      --checkpoint "$checkpoint" --manifest "$VAL_MANIFEST" --stats-dir "$STATS" \
      --text-cache "$TEXT" --protocol in_horizon --total-seconds 9 \
      --selection-min-seconds 9 --samples 64 --batch-size 2 --diffusion-steps 10 \
      --cfg-weight-text 2 --cfg-weight-constraint 2 --seed 20260831 \
      --device cuda:0 --panel "$PANEL_ROOT/9s_in_horizon.jsonl" \
      --protocol-lock "$PROTOCOL_LOCK" --attention-window-chunks 4 \
      --output "$out/9s_in_horizon/result.json" \
      --per-sample-output "$out/9s_in_horizon/per_sample.jsonl" \
      --state-diagnostics-output "$out/9s_in_horizon/state.jsonl" \
      >"$out/9s_in_horizon/eval.log" 2>&1
  ) &
  local long_pid=$!
  local status=0
  wait "$table_pid" || status=1
  wait "$long_pid" || status=1
  return "$status"
}

validate_health_gate() {
  local target=$1 stage=$2
  "$PY" - "$target" "$stage" "$NATIVE_HEALTH" <<'PY'
import json
import math
import sys
from pathlib import Path

target = int(sys.argv[1])
stage = Path(sys.argv[2])
native = Path(sys.argv[3])
metrics = ("mobench_root_ade_cm_mean", "constraint_error_cm_mean")
checks = {}
values = {}
for variant in ("full", "no_delta"):
    checks[variant] = {}
    values[variant] = {}
    for panel in ("table4", "9s_in_horizon"):
        baseline = json.loads((native / panel / "result.json").read_text())
        candidate = json.loads(
            (stage / "health" / variant / panel / "result.json").read_text()
        )
        same_panel = (
            baseline.get("panel_sha256") == candidate.get("panel_sha256")
            and baseline.get("sample_ids") == candidate.get("sample_ids")
        )
        panel_checks = {"same_registered_panel": same_panel}
        panel_values = {}
        for metric in metrics:
            before = float(baseline[metric])
            after = float(candidate[metric])
            degradation = (after - before) / max(abs(before), 1e-12)
            panel_checks[f"{metric}_finite"] = math.isfinite(after)
            panel_checks[f"{metric}_relative_degradation_le_5_percent"] = degradation <= 0.05
            panel_values[metric] = {
                "native": before,
                "candidate": after,
                "relative_degradation": degradation,
            }
        checks[variant][panel] = panel_checks
        values[variant][panel] = panel_values
passed = all(
    value
    for variant in checks.values()
    for panel in variant.values()
    for value in panel.values()
)
result = {
    "status": "PASS" if passed else "FAIL",
    "target_total_optimizer_steps": target,
    "threshold": "relative degradation <= 5% versus registered Native Hybrid",
    "checks": checks,
    "values": values,
    "locked_final_actors_read": False,
}
(stage / "health_gate.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
raise SystemExit(0 if passed else 1)
PY
}

for target in "${TARGETS[@]}"; do
  echo "[$(date -Is)] starting D2R-L target step $target"
  run_pair_stage "$target"
  echo "[$(date -Is)] completed D2R-L target step $target"
done

date -Is >"$RUN_ROOT/LONG_TRAINING_COMPLETE"
echo "[$(date -Is)] D2R-L long training completed"
