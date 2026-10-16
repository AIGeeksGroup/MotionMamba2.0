"""Reusable minimal/hybrid sequence kernel.

The package itself is Mamba-free on import. The existing Mamba wrapper is
loaded only when a hybrid :class:`SequenceKernel` is constructed.
"""

from .config import SequenceKernelConfig
from .operator import SequenceKernel
from .state import PendingSequenceKernelState, SequenceKernelState
from .token_state import TokenStateInteraction, TokenStateVariant

__all__ = [
    "SequenceKernelConfig",
    "SequenceKernel",
    "SequenceKernelState",
    "PendingSequenceKernelState",
    "TokenStateInteraction",
    "TokenStateVariant",
]
