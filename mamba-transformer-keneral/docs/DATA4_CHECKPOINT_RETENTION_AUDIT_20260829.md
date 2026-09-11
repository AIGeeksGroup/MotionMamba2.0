# DATA4 Checkpoint Retention and Reproduction Audit (2026-08-29)

## Scope

This cleanup removes obsolete experiment checkpoints from
`/data4/zts/result` while retaining datasets, manifests, normalization
statistics, text caches, training/evaluation logs, JSON metrics, NPZ evaluation
artifacts, and the checkpoints required for current results and future
training. It does not modify `/data4/zts/datasets`.

The cleanup is implemented by
`examples/cleanup_data4_checkpoints_20260829.sh`. The script defaults to a dry
run, validates every retained file before deletion, records every removed path,
and hashes all retained checkpoints after deletion.

## Retained Checkpoint Contract

| Group | Retained checkpoints | Purpose |
|---|---:|---|
| BONES-SEED healthy 50k | ARDY, Minimal, Hybrid full resume + inference weights | Stable starting point for future unified training |
| BONES-SEED healthy ARDY retrain | final 50k resume + inference weights | ARDY generated-history starting point |
| BONES-SEED v2 step900 | ARDY, Minimal, Hybrid full resume + inference weights | Fixed baseline behind the final exact-panel Table 4/5 comparison |
| BONES-SEED v9 step100 | Minimal and Hybrid full resume checkpoints | Current best independent-validation checkpoint |
| HumanML3D healthy 50k | ARDY, Minimal, Hybrid full resume + inference weights | Stable starting point for future HumanML3D training |

The authoritative retained paths and SHA-256 hashes are written to:

- `/data4/zts/result/checkpoint_cleanup_20260829/retained_paths.txt`
- `/data4/zts/result/checkpoint_cleanup_20260829/retained_checkpoints.sha256`
- `docs/DATA4_RETAINED_CHECKPOINTS_20260829.sha256` (repository snapshot)

Every deleted checkpoint path, logical size, inode, and pre-deletion link count
is recorded in
`/data4/zts/result/checkpoint_cleanup_20260829/deleted_checkpoint_paths.tsv`.

## Reproduce Current v9

The v9 step100 training and actor-disjoint validation are reproduced with:

```bash
bash examples/run_bones_seed_endpoint_v9_stage100_reproduce_20260829.sh
```

This launcher fixes the generated-history probability schedule, deep-v3
rollout distribution, four-window accepted-history BPTT, endpoint loss,
short-window replay, Mamba readout decay, optimizer, batch size, and random
seed used by the retained checkpoint.

The exact v2 test-panel Table 4/5 comparison is reproduced with:

```bash
bash examples/run_bones_seed_v9_v2_test_protocol_20260829.sh
```

The latter asserts ordered sample IDs and all shared protocol fields against
the v2 panel before writing
`docs/BONES_SEED_V9_V2_EXACT_TEST_PROTOCOL_COMPARISON_20260829.md`.

## Reproduce Healthy Starting Points

BONES-SEED three-model 50k training:

```bash
bash examples/run_bones_seed_contract_full50k_20260824.sh
```

Healthy ARDY retraining:

```bash
bash examples/run_bones_seed_ardy_retrain_20260826.sh
```

HumanML3D three-model 50k training:

```bash
bash examples/run_cache_contract_full50k_20260823.sh
```

## Final Evaluation Artifacts

- Exact v2-panel comparison:
  `/data4/zts/result/bones_seed_generated_history_endpoint_v9_v2_test_protocol_20260829`
- Independent actor-disjoint v9 validation:
  `/data4/zts/result/bones_seed_generated_history_endpoint_v9_val_stage100_20260829`
- Main training audit: `docs/BONES_SEED_EXPOSURE_GAP_TRAINING_AUDIT_20260829.md`
- Complete Table 4/5 comparison:
  `docs/BONES_SEED_V9_V2_EXACT_TEST_PROTOCOL_COMPARISON_20260829.md`

Old logs and lightweight metrics remain available for historical diagnosis,
but their large intermediate checkpoints are intentionally not retained.

