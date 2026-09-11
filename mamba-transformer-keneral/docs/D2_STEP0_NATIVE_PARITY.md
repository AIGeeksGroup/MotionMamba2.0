# D2 Step-0 Native Hybrid Parity

Date: 2026-09-03  
Status: PASS

The healthy BONES Native Hybrid 50k parent was verified against
`docs/B_BASELINE_MANIFEST.json` with SHA-256
`cfa9d44d9a804e3f71dbfde98cbad7faceed98bca83b5417c3afb5bca6071a4a`.
Native and D2 used the same fixed selection sample
`injured_leg_jog_ff_180_R_001__A543`, seed `9032026`, timestep/noise, text,
constraints, and weights.

| Check | Difference |
|---|---:|
| Combined raw denoiser output max abs | 0 |
| Root block raw output max abs | 0 |
| Body block raw output max abs | 0 |
| One-window denoising loss abs | 0 |
| Root Attention committed KV max abs | 0 |
| Body Attention committed KV max abs | 0 |
| Root Mamba committed state max abs | 0 |
| Body Mamba committed state max abs | 0 |
| 52-frame geometry proxy max abs | 0 |

The root/body Attention and Mamba SHA-256 values also match exactly. At the
same time, the D2 accepted write produced nonzero slots (mean norm `0.163756`)
with write-gate mean `0.499585`. The read residual and state-query effective
scales were both exactly zero. This verifies that the new branch is alive and
stateful while preserving the trained Native function at migration step 0.

Raw result:
`/data4/zts/result/stage_d2_step0_native_parity_20260903/results.json`
(`SHA-256 3d1c882c32202c6a84f55d7c5f5415157bd11bbaa2647a7dcbd9966d8087df04`).

