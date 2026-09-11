# Reproduction Commands

Run from `mamba-transformer-keneral` with the recorded environment:

```bash
export PYTHONPATH=src:/data3/zts/packages/state-spaces-mamba:/data3/zts/packages/mamba_op_triton
examples/run_stage_c_reproduction_20260902.sh
examples/run_stage_d_contract_tests_20260902.sh
```

The first command reproduces the historical C gate when its recorded external
checkpoint/cache are restored. The second command runs the local D contract
suite; current result: **58 passed**.

No E-I command is supplied because their registered gates were not met.
