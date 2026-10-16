"""Tunable configuration for the State-Transition Memory Operator.

Every knob a downstream user (e.g. an ARDY denoiser block) is expected to
adjust lives here. Defaults match the values validated in the research repo
(`mamba_algorithm`) on synthetic recall tasks (MQAR, repeated_reference) --
see `docs/mamba_integration_contract.md` for what's been validated and what
hasn't.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StateMemoryOperatorConfig:
    d_model: int
    """Must match the Mamba3 instance's `d_model`."""

    d_summary: int = 128
    """Width of the compressed state summary StateProjector produces from
    the raw `ssm_state`."""

    d_qkv: int = 64
    """Width of the state-derived Q/K vectors used for memory addressing."""

    d_value: int = 64
    """Width of the content-derived V vectors stored in memory. Validated
    finding (2026-07-21/22): deriving V from state (like Q/K) does NOT work
    -- a linear probe over the full state tops out under 1% decodability for
    fine-grained content. V must be a projection of the chunk's own content
    representation, not the state."""

    memory_capacity: int = 64
    """Fixed slot count per batch row. FIFO shows a hard accuracy cliff the
    instant a write falls outside this window, regardless of how often it's
    later queried (validated 2026-07-25) -- size this to the longest horizon
    you need reliable recall over, or use `memory_backend="participative"`
    to extend effective retention beyond raw capacity for frequently
    -reaccessed content."""

    topk: int = 4
    """Retrieval candidates blended via softmax. Validated finding
    (2026-07-25): a HARD single-candidate read (`topk=1`) is a WORSE
    training signal than soft blending, even when the single candidate is
    guaranteed correct -- prefer `topk >= 2` unless you have a specific
    reason not to."""

    retrieval_temperature: float = 0.1

    memory_backend: str = "fifo"
    """"fifo": evict oldest write when a row exceeds capacity.
    "participative": evict lowest EMA-decayed participation score instead --
    validated (2026-07-25) to eliminate the FIFO capacity cliff for content
    that gets re-queried, at no measurable accuracy cost on the
    within-capacity case, and to be MORE seed-stable than FIFO."""

    participation_decay: float = 0.98
    """EMA decay for the participative backend's participation score."""

    injection_mode: str = "input"
    """"input": inject the memory correction into the chunk's input before
    the Mamba recurrence runs. "output": inject after the recurrence,
    directly into the readout, right before any downstream head. Validated
    finding (2026-07-23, frozen-backbone setting only): if the backbone is
    FROZEN, "output" injection dramatically outperforms "input" (~10.8% vs
    ~0% within-capacity) because a frozen recurrence never learns to
    interpret an injected input. If the backbone is jointly trainable, the
    two modes performed identically in every test run so far -- "input" is
    the default only because it's what most experiments used, not because
    it's been shown better."""

    use_content_query: bool = True
    """Mixes a projection of the chunk's OWN content into the query vector
    used for retrieval, alongside the state-derived query. Needed whenever
    a single forward_chunk call must be able to distinguish "which query is
    this" using information the pre-chunk state cannot know (e.g. this
    chunk's own key/identity). Set False only if your read timing makes
    this unnecessary."""

    pooling: str = "mean"
    """StateProjector's pooling over `ssm_state`'s `d_state` axis: "mean" or
    "rms". Only "mean" has been validated end-to-end so far."""
