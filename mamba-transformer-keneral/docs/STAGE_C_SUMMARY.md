# Stage C Summary

Date: 2026-09-02
Status: FAIL (historical real-cache 100-step gate)

Stage C was already executed on a real sequential teacher cache and is recorded
in [stage06_contribution1_closeout.md](stage06_contribution1_closeout.md). This
stage result deliberately preserves that result rather than rerunning against a
different, unverified cache contract.

| Variant | State delta | First loss | Final loss | Reduction | Final eval | Gate |
|---|---:|---:|---:|---:|---:|---|
| `state_query` | no | 0.6960473 | 0.6678084 | 4.06% | 0.8190935 | FAIL |
| `state_query` | yes | 0.6960473 | 0.6684049 | 3.97% | 0.7930969 | FAIL |

Both runs used the existing `StateProjector -> state_summary -> state_delta ->
state_to_address` path, neutral initialization, accepted clean history, bf16
Mamba and fp32 attention. Gradients were finite, but neither variant
substantially overfit one fixed real case and the delta variant did not improve
on state-only. The naive and state-KV definitive jobs were not completed due to
external GPU memory contention; they cannot be counted as a pass.

The 100-step gate therefore fails. The required 250-step step-up and all
generated 20/40-second C evaluations are intentionally not run. Per the
registered route, this directs work to D's durable content mechanism; it does
not authorize E, F, G, H, or I.

Reproduction inputs from the historical run are no longer mounted at their
recorded `/workspace/ardy_stage05` paths. The exact command is retained in
`examples/run_stage_c_reproduction_20260902.sh`; a new run requires restoring
that checkpoint and cache with their recorded hashes.
