# D3c External Read Contract

`external_memory_level = read_ready`.

The hook is enabled only for noisy diffusion reads after the requested
`read_epoch`. It supplies the exact `[B,K,D]` tensor that normal
`PersistentContentMemory.read(...).to(x.dtype)` would have returned. The
recipient computes memory K/V with its own current layer projections. The hook
never mutates durable slots, Mamba state, state summary/delta, Attention W4
KV, write proposal, or accepted clean commit. At a clean accepted commit the
view is cleared before `canonical_history_prime`.

Every external tensor must match the recipient read-ready tensor's shape,
dtype, device, stride and contiguity. CFG expansion is outside this contract:
mapping is resolved at the user batch size before ARDY expands guidance rows.
