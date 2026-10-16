"""State-Transition Memory Operator: clean, ready-to-use package.

    from state_memory_operator import (
        StateMemoryOperatorConfig,
        StateTransitionMemoryOperator,
        OperatorState,
        EventMemory,
        Mamba3ChunkWrapper,
        Mamba3State,
    )

See `examples/quickstart.py` for end-to-end usage and
`docs/mamba_integration_contract.md` for what's validated vs. not yet.
"""

from importlib import import_module

from .config import StateMemoryOperatorConfig
from .event_memory import EventMemory
from .content_memory import ContentMemoryPending, PersistentContentMemory
from .state_qkv import ContentQueryEncoder, ContentValueEncoder, StateQKVEncoder


_LAZY_EXPORTS = {
    "Mamba3ChunkWrapper": (".mamba3_chunk_wrapper", "Mamba3ChunkWrapper"),
    "Mamba3State": (".mamba3_chunk_wrapper", "Mamba3State"),
    "apply_reset_mask": (".mamba3_chunk_wrapper", "apply_reset_mask"),
    "OperatorState": (".operator", "OperatorState"),
    "StateTransitionMemoryOperator": (".operator", "StateTransitionMemoryOperator"),
    "StateProjector": (".state_projector", "StateProjector"),
    "StateTransition": (".state_projector", "StateTransition"),
    "project_transition": (".state_projector", "project_transition"),
}


def __getattr__(name: str):
    """Load Mamba-dependent exports only when a caller asks for one."""

    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

__all__ = [
    "StateMemoryOperatorConfig",
    "StateTransitionMemoryOperator",
    "OperatorState",
    "EventMemory",
    "PersistentContentMemory",
    "ContentMemoryPending",
    "Mamba3ChunkWrapper",
    "Mamba3State",
    "apply_reset_mask",
    "StateProjector",
    "StateTransition",
    "project_transition",
    "StateQKVEncoder",
    "ContentValueEncoder",
    "ContentQueryEncoder",
]
