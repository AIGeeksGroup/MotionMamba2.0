# Stage 06R Closeout — State-Guided Routing over Attention KV

Date: 2026-08-14
Decision: stopped at failed Gate C

## 1. Sequential evaluation dataset

Gate A used the frozen Stage-05 eval cache:

```text
/workspace/ardy_stage05/cache/minimal_seq/manifest_eval.json
```

It contains 16 real teacher trajectories, four accepted 40-frame chunks, ten
diffusion calls per chunk, and replay error exactly zero. First/middle/final
diffusion calls for current chunks 1--3 produced 144 cases. The 96 cases at
chunks 2--3 have at least two historical candidates and are the nontrivial
routing set. Router training used the corresponding 64-trajectory train cache
(384 nontrivial cases) and the 16-trajectory eval cache (96 held-out cases).

The frozen parent was
`/workspace/ardy_stage05/outputs/minimal_real_v1/sequence_minimal.pt`, SHA-256
`acbdcc78c6594ad690a1d3a76b129665198fd142e487a1eb64bf43163a1b949f`.

## 2. Oracle definition

For current chunk `t`, candidates were accepted clean chunks before `t`.
Sinks and current K/V were always retained. Frozen Minimal was replayed for no
history, Recent, Oldest, Full, every singleton, and every two-chunk subset.
Oracle Top-1/Top-2 are the singleton/pair with minimum root+body denoising MSE.
No Mamba state or future chunk was used to construct an oracle label.

## 3. Oracle Full versus Top-1/Top-2

Across the 96 nontrivial eval cases:

```text
Full             mean MSE 0.2325568
Oracle Top-1     mean MSE 0.2321406
Oracle Top-2     mean MSE 0.2318964
```

The important stratum was current chunk 3, where three candidates exist:

```text
Full             0.1101033
Oracle Top-1     0.1098429
Oracle Top-2     0.1087825
```

Oracle Top-2 improved chunk-3 MSE by 1.07% on average. A trajectory-level
10,000-sample bootstrap gave absolute-gain 95% CI
`[0.000438, 0.002336]`; 12/16 trajectories had positive mean gain.

## 4. Oracle selected-distance distribution

For the 96 nontrivial Top-1 labels:

```text
distance 1: 49
distance 2: 28
distance 3: 19
```

Only 51.04% selected the most recent chunk. At chunk 3, 24/48 labels were
non-recent, so the oracle was not a disguised Recent heuristic.

## 5. Gate A decision

`PASS`, narrowly and specifically for Top-2. Oracle Top-1 was not globally
robust, particularly at the highest-noise diffusion call. The supported claim
was only that selecting two of three chunks can improve frozen denoising on
this real cache.

## 6. Router architecture

Frozen accepted history was processed by existing Mamba3 and StateProjector.
Across root/body and all DiT layers, committed summaries formed `s_(t-1)` and
candidate summaries were mean-pooled existing attention K. Routers used:

```text
q = normalize(Wq source)
k_j = normalize(Wk mean_K(C_j))
score_j = q dot k_j / 0.1
```

Sources were current noisy root/body token means (token-only), committed state
(state-only), or committed state concatenated with temporal state delta. Only
`Wq/Wk` trained; Minimal, attention, Mamba3, and StateProjector were frozen.

No trained real-motion Mamba checkpoint exists. Mamba3 and StateProjector
therefore used deterministic random initialization with seed 79. Failed
Stage-06 pilot weights were intentionally not reused as pretraining.

## 7. Router tiny-overfit

All variants passed the 32-example, middle-diffusion tiny-overfit gate:

```text
token       loss 1.04372 -> 0.000760, Hit@1 1.0
state       loss 0.84472 -> 0.000398, Hit@1 1.0
state+delta loss 1.10828 -> 0.000541, Hit@1 1.0
```

Gradients were finite and only router projections were optimized.

## 8. Recent baseline

Held-out nontrivial routing metrics:

```text
Hit@1 0.5104
Hit@2 0.8021
MRR   0.7222
```

## 9. Token-only router

```text
train Hit@1 0.9583
eval loss   0.78385
eval Hit@1  0.7813
eval Hit@2  0.9375
eval MRR    0.8802
```

On the nontrivial three-candidate chunk-3 stratum, Hit@2 was 0.8750.

## 10. State-only router

```text
train Hit@1 0.7214
eval loss   0.72183
eval Hit@1  0.5729
eval Hit@2  0.9792
eval MRR    0.7830
```

On chunk 3, Hit@2 was 0.9583 (46/48), versus 0.8750 for token-only. State
lost on Hit@1 and MRR but supplied stronger Top-2 recall, the use case Gate A
had supported.

## 11. State plus delta router

```text
train Hit@1 0.7188
eval loss   0.75204
eval Hit@1  0.5729
eval Hit@2  0.9792
eval MRR    0.7830
```

State+delta gave exactly the same aggregate Hit@1/Hit@2/MRR as state-only and
worse loss. It is not supported as an incremental contribution.

## 12. Distance-stratified router metrics

State-only Hit@1 for oracle distance 1/2/3 was
`0.5102 / 0.4643 / 0.8947`; token-only was
`0.7551 / 0.6786 / 1.0000`. State's advantage was Top-2 coverage rather than
best-chunk identification. State+delta shifted errors between distances 2 and
3 without improving aggregate metrics.

## 13. Gate B decision

`PASS` only for frozen Top-2 routing. State Top-2 beat Recent and token-only on
the true three-candidate cases. Gate B does not support a Top-1 claim, does not
support state over token on Hit@1/MRR, and does not support delta.

## 14. Frozen-model denoising results

Exact previously replayed Minimal subset losses were indexed by learned router
rankings; no generator weight changed.

```text
policy                 all 96 MSE    chunk-3 MSE
Full                    0.2325568      0.1101033
Recent Top-2            0.2331879      0.1113656
Token Top-2             0.2332362      0.1114620
State Top-2             0.2327491      0.1104880
State+delta Top-2       0.2327491      0.1104880
Oracle Top-2            0.2318964      0.1087825
```

State Top-2 was better than learned token and Recent Top-2, but worse than
Full. For chunk 3 its paired Full-minus-State gain was `-0.000385`; the
trajectory-bootstrap 95% CI was `[-0.001298, 0.000287]`, with only 6/16
trajectories positive.

## 15. Full W / Top-2 / Top-1 comparison

State Top-1 was not viable: overall MSE `0.2345536` versus Full `0.2325568`.
State Top-2 closed much of that gap but remained worse than Full. Oracle Top-2
showed that a better selector could improve quality; the learned frozen state
selector did not realize that oracle gain.

## 16. Latency, memory, and K/V-token counts

Each 40-frame chunk maps to ten motion tokens. At chunk 3:

```text
policy       history KV   total KV including 4 sinks/current   attention ratio
Full              30                      44                       1.0000
Top-2             20                      34                       0.7727
Top-1             10                      24                       0.5455
```

Gate-A frozen inference consumed 66.30 seconds for 144 cases and peaked at
2201.84 MiB. Initial Gate-B Mamba feature extraction plus router fitting peaked
at 2433.66 MiB; cached router-only rerun peaked at 228.31 MiB.

Per-policy end-to-end latency and peak memory were not benchmarked because
Gate C failed quality improvement and a real state policy must include Mamba
control overhead. Reporting attention-only token-count savings as end-to-end
speedup would be misleading. No fused-kernel speedup is claimed.

## 17. Gate C decision

`FAIL`. Learned state Top-2 did not lower denoising error versus Full. Although
it removed one of three history chunks, net efficiency was not established
after accounting for the required Mamba/state-router path. The experiment
therefore stopped before generator training or routing integration.

## 18. Exact interpretation

The real cache supports selective-history value (Oracle Top-2) and shows that
the frozen state representation can recover the oracle-best singleton within
its Top-2 more often than the token-only router. It does not show that this
ranking improves the frozen denoiser over Full, and it does not establish a
MotionMamba Contribution 1.

“Token-state delta” remains ambiguous: temporal state delta, temporal token
delta, and token-state discrepancy are distinct ideas. Only temporal state
delta was tested here, as an ablation, and it added no value. The other two
were not implemented.

## 19. Recommended next step

Do not scale or train this router. If the direction is revisited, first obtain
a real-motion-trained Mamba state encoder or train a state representation with
explicit routing supervision while keeping the generator frozen, then repeat
Gates B and C. That would be a new bounded experiment, not permission to train
full MotionMamba or begin Contribution 2.

Artifacts:

```text
/workspace/ardy_stage06r/gate_a_oracle/oracle_results.json
/workspace/ardy_stage06r/gate_b_oracle_train/oracle_results.json
/workspace/ardy_stage06r/gate_b_router/router_features.pt
/workspace/ardy_stage06r/gate_b_router/router_results.json
/workspace/ardy_stage06r/gate_c_frozen_routing.json
```

C1_ROUTING_STOP
