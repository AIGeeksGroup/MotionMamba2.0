#!/usr/bin/env bash
set -euo pipefail
ROOT=/data3/zts/MotionMamba2.0/mamba-transformer-keneral
PY=/data3/zts/conda_envs/ardy_env/bin/python
BASE=/data4/zts/result
OUT=$BASE/stage_d2rl_frozen_5k_final_eval_20260906
PANEL=$BASE/bones_seed_long_memory_panel_v2_20260831/selection
LOCK=$ROOT/configs/bones_seed_long_memory_protocol_v2.json
MANIFEST=$BASE/bones_seed_ardy_val/manifest.jsonl
STATS=$BASE/bones_seed_ardy_train
TEXT=$BASE/bones_seed_text_cache_llm2vec
D2=$BASE/stage_d2rl_bones_long_20260904/step5000/full/hybrid_humanml3d_ddp.pt
NATIVE=$BASE/bones_seed_contract_full50k_20260824_hybrid/hybrid_humanml3d_ddp.pt
MINIMAL=$BASE/bones_seed_contract_full50k_20260824_minimal/minimal_humanml3d_ddp.pt
ARDY=$BASE/bones_seed_ardy_retrain_lr1e5_20260826_original/original_humanml3d_ddp.pt
SQ=$BASE/stage_d2rl_fair_ablations_5k_20260906/state_query_only/hybrid_humanml3d_ddp.pt
MEM=$BASE/stage_d2rl_fair_ablations_5k_20260906/persistent_memory_only_v3/hybrid_humanml3d_ddp.pt
export HF_HOME=/data3/zts/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LOCAL_CACHE=true
export PYTHONPATH=$ROOT/src:/data3/zts/MotionMamba2.0/ardy:/data3/zts/packages/state-spaces-mamba:/data3/zts/packages/mamba_op_triton${PYTHONPATH:+:$PYTHONPATH}
while [[ ! -s "$SQ" || ! -s "$MEM" ]]; do sleep 60; done
mkdir -p "$OUT"

run_table4() {
  local gpu=$1 name=$2 mode=$3 ckpt=$4 dir="$OUT/$2/table4"
  [[ -f "$dir/result.json" ]] && return 0; mkdir -p "$dir"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$ROOT/examples/evaluate_humanml3d_table45.py" \
    --mode "$mode" --model g1 --dataset-name BONES-SEED-v004 --checkpoint "$ckpt" \
    --manifest "$MANIFEST" --stats-dir "$STATS" --text-cache "$TEXT" --protocol table4 \
    --samples 64 --batch-size 2 --diffusion-steps 10 --cfg-weight-text 2 --cfg-weight-constraint 2 \
    --seed 20260831 --device cuda:0 --panel "$PANEL/table4.jsonl" --protocol-lock "$LOCK" \
    --attention-window-chunks 4 --output "$dir/result.json" --per-sample-output "$dir/per_sample.jsonl" \
    >"$dir/eval.log" 2>&1
}
run_long() {
  local gpu=$1 name=$2 mode=$3 ckpt=$4 sec=$5 proto=$6 intervention=${7:-normal}
  local suffix=""; [[ "$intervention" != normal ]] && suffix="__${intervention}"
  local dir="$OUT/$name/${sec}s_${proto}${suffix}"; [[ -f "$dir/result.json" ]] && return 0; mkdir -p "$dir"
  local samples=16 sel=$sec; [[ "$sec" == 9 ]] && samples=64; [[ "$sec" == 20 ]] && sel=40
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$ROOT/examples/evaluate_ardy_official_table5.py" \
    --mode "$mode" --model g1 --dataset-name BONES-SEED-v004 --checkpoint "$ckpt" \
    --manifest "$MANIFEST" --stats-dir "$STATS" --text-cache "$TEXT" --protocol "$proto" \
    --total-seconds "$sec" --selection-min-seconds "$sel" --samples "$samples" --batch-size 2 \
    --diffusion-steps 10 --cfg-weight-text 2 --cfg-weight-constraint 2 --seed 20260831 --device cuda:0 \
    --panel "$PANEL/${sec}s_${proto}.jsonl" --protocol-lock "$LOCK" --attention-window-chunks 4 \
    --content-memory-intervention "$intervention" --output "$dir/result.json" --per-sample-output "$dir/per_sample.jsonl" \
    --state-diagnostics-output "$dir/state.jsonl" >"$dir/eval.log" 2>&1
}
wave() { local status=0 p; for spec in "$@"; do eval "$spec" & done; for p in $(jobs -pr); do wait "$p" || status=1; done; return "$status"; }
wave 'run_table4 0 d2rl_5k hybrid "$D2"' 'run_table4 1 native_hybrid hybrid "$NATIVE"' 'run_table4 2 minimal_w4 minimal "$MINIMAL"' 'run_table4 3 ardy original "$ARDY"' 'run_table4 4 state_query_only hybrid "$SQ"' 'run_table4 5 persistent_memory_only hybrid "$MEM"'
wave 'run_long 0 d2rl_5k hybrid "$D2" 9 in_horizon' 'run_long 1 native_hybrid hybrid "$NATIVE" 9 in_horizon' 'run_long 2 minimal_w4 minimal "$MINIMAL" 9 in_horizon' 'run_long 3 ardy original "$ARDY" 9 in_horizon' 'run_long 4 d2rl_5k hybrid "$D2" 9 out_of_horizon' 'run_long 5 native_hybrid hybrid "$NATIVE" 9 out_of_horizon' 'run_long 6 minimal_w4 minimal "$MINIMAL" 9 out_of_horizon' 'run_long 7 ardy original "$ARDY" 9 out_of_horizon'
wave 'run_long 0 d2rl_5k hybrid "$D2" 20 in_horizon' 'run_long 1 native_hybrid hybrid "$NATIVE" 20 in_horizon' 'run_long 2 minimal_w4 minimal "$MINIMAL" 20 in_horizon' 'run_long 3 ardy original "$ARDY" 20 in_horizon' 'run_long 4 d2rl_5k hybrid "$D2" 40 in_horizon' 'run_long 5 native_hybrid hybrid "$NATIVE" 40 in_horizon' 'run_long 6 minimal_w4 minimal "$MINIMAL" 40 in_horizon' 'run_long 7 ardy original "$ARDY" 40 in_horizon'
wave 'run_long 0 state_query_only hybrid "$SQ" 40 in_horizon' 'run_long 1 persistent_memory_only hybrid "$MEM" 40 in_horizon' 'run_long 2 d2rl_5k hybrid "$D2" 40 in_horizon write_off' 'run_long 3 d2rl_5k hybrid "$D2" 40 in_horizon readout_off'
ROOT="$ROOT" OUT="$OUT" "$PY" "$ROOT/examples/write_d2rl_frozen_5k_reports_20260906.py"
date -Is > "$OUT/FINAL_EVALUATION_COMPLETE"
