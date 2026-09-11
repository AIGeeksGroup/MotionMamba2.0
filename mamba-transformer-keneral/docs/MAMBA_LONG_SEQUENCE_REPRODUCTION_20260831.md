# Mamba Long-Sequence Reproduction (2026-08-31)

Run from `/data3/zts/MotionMamba2.0/mamba-transformer-keneral`. Long GPU jobs
must use `tmux`; launchers are idempotent and write under `/data4/zts/result`.

## Preflight

```bash
cd /data3/zts/MotionMamba2.0/mamba-transformer-keneral
PY=/data3/zts/conda_envs/ardy_env/bin/python
export HF_HOME=/data3/zts/cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LOCAL_CACHE=true
export PYTHONPATH=$PWD/src:/data3/zts/MotionMamba2.0/ardy:/data3/zts/packages/state-spaces-mamba:/data3/zts/packages/mamba_op_triton${PYTHONPATH:+:$PYTHONPATH}
$PY -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))'
test -f /data4/zts/result/bones_seed_ardy_train/manifest.jsonl
test -f /data4/zts/result/bones_seed_ardy_val/manifest.jsonl
test -f /data4/zts/result/bones_seed_ardy_test/manifest.jsonl
```

## Protocol And Panels

The checked-in v2 lock is authoritative. Rebuild to a new directory, then
compare all actor/manifest/panel hashes with the lock:

```bash
$PY examples/build_bones_seed_long_memory_panel.py \
  --train-manifest /data4/zts/result/bones_seed_ardy_train/manifest.jsonl \
  --selection-manifest /data4/zts/result/bones_seed_ardy_val/manifest.jsonl \
  --locked-final-manifest /data4/zts/result/bones_seed_ardy_test/manifest.jsonl \
  --text-cache /data4/zts/result/bones_seed_text_cache_llm2vec \
  --output-dir /data4/zts/result/bones_seed_long_memory_panel_v2_reproduction_20260831 \
  --seed 20260831 \
  --protocol-output /data4/zts/result/bones_seed_long_memory_panel_v2_reproduction_20260831/protocol.json
```

Do not evaluate `locked_final/*` during configuration or checkpoint selection.

## Tests

```bash
$PY -m compileall -q src examples benchmarks tests
$PY -m pytest -q
CUDA_VISIBLE_DEVICES=3 bash examples/run_differentiable_rollout_cuda_test_20260831.sh
```

Recorded: `115 passed, 6 skipped`; CUDA test `1 passed`.

## Stage A

```bash
tmux new-session -d -s bones_longmem_stage_a_repro \
  'bash /data3/zts/MotionMamba2.0/mamba-transformer-keneral/examples/run_bones_seed_long_memory_stage_a_20260831.sh'
tmux attach -t bones_longmem_stage_a_repro
```

Expected marker:
`/data4/zts/result/bones_seed_long_memory_stage_a_20260831/STAGE_A_COMPLETE`.

## Stage B Training

ARDY, Minimal W4/W8/W16/ParamMatch, and no-decay Hybrid ablations:

```bash
tmux new-session -d -s bones_longmem_stage_b_repro \
  'bash /data3/zts/MotionMamba2.0/mamba-transformer-keneral/examples/run_bones_seed_long_memory_stage_b_20260831.sh'
```

Registered Hybrid decay start W4, half-life 4, minimum 0:

```bash
tmux new-session -d -s bones_longmem_stage_b_decay_repro \
  'bash /data3/zts/MotionMamba2.0/mamba-transformer-keneral/examples/run_bones_seed_long_memory_stage_b_decay_20260831.sh'
```

```bash
test -f /data4/zts/result/bones_seed_long_memory_stage_b_20260831/STAGE_B_TRAINING_COMPLETE
test -f /data4/zts/result/bones_seed_long_memory_stage_b_decay_20260831/STAGE_B_DECAY_TRAINING_COMPLETE
```

Resume contracts reject changes to rollout, losses, Attention window,
ParamMatch width, state intervention, seed/batch/LR, and decay fields.

## Selection Evaluation And Gate

```bash
tmux new-session -d -s bones_longmem_stage_b_eval_repro \
  'bash /data3/zts/MotionMamba2.0/mamba-transformer-keneral/examples/run_bones_seed_long_memory_stage_b_selection_eval_20260831.sh'
```

The launcher consumes only immutable `selection/*` panels. It writes aggregate
JSON, per-sample JSONL, state/intervention JSONL, and NPZ for stage 0/100/250
Table 4 and 9 seconds, stage 100/250 20/40 seconds, capacity/context/decay
controls, and same-checkpoint interventions.

Main output:
`/data4/zts/result/bones_seed_long_memory_stage_b_selection_eval_20260831`.

To recompute paired statistics without regenerating motion:

```bash
$PY examples/check_bones_seed_long_memory_stage_b_gate.py \
  --evaluation-root /data4/zts/result/bones_seed_long_memory_stage_b_selection_eval_20260831 \
  --training-root /data4/zts/result/bones_seed_long_memory_stage_b_decay_20260831 \
  --output /data4/zts/result/bones_seed_long_memory_stage_b_selection_eval_20260831/gate_summary.reproduced.json
```

The recorded gate is `FAIL` with `next_stage_authorized=false`. Do not run 1k,
5k, or locked-final for this experiment. In a new preregistered run, 1k requires
both 100 and 250 to pass, must use seeds `827321, 827322, 827323`, and must cover
ARDY, Minimal-W4, Minimal-ParamMatch, Hybrid carry/fresh, and at least W8. A 5k
stage requires the three-seed 1k aggregate to pass. Locked-final additionally
requires frozen config/checkpoint/panel hashes and a one-time marker, and must
not be described as a never-observed blind test.

## System Benchmark

Run only after GPU training and quality evaluation end:

```bash
tmux new-session -d -s bones_longmem_scaling_repro \
  'bash /data3/zts/MotionMamba2.0/mamba-transformer-keneral/examples/run_ardy_long_sequence_scaling_20260831.sh'
```

Outputs are in
`/data4/zts/result/bones_seed_long_sequence_scaling_v2_20260831`. This is a
real-checkpoint root-block benchmark, not an end-to-end quality/latency claim.

## Final Bundle

```bash
$PY examples/write_mamba_long_sequence_evidence_bundle_20260831.py
```

This writes `summary.json`, `ablation_ledger.jsonl`, and `SHA256SUMS` under
`/data4/zts/result/mamba_long_sequence_evidence_20260831`.
