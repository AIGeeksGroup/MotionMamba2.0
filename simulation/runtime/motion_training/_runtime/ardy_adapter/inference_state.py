"""Ephemeral CFG expansion: physical copies, never persistent/session state."""
import copy
from dataclasses import fields, is_dataclass
import torch

def expand_cfg_state(value, groups):
    if isinstance(value, torch.Tensor):
        return value.repeat((groups,) + (1,) * (value.ndim - 1)).contiguous()
    if is_dataclass(value):
        return type(value)(**{f.name: expand_cfg_state(getattr(value, f.name), groups)
                              for f in fields(value)})
    if isinstance(value, list): return [expand_cfg_state(v, groups) for v in value]
    if isinstance(value, tuple): return tuple(expand_cfg_state(v, groups) for v in value)
    if isinstance(value, dict): return {k: expand_cfg_state(v, groups) for k,v in value.items()}
    if hasattr(value, '__dict__'):
        result=copy.copy(value)
        for k,v in vars(value).items():
            setattr(result,k,v*groups if k=='batch_size' else expand_cfg_state(v,groups))
        return result
    return value
