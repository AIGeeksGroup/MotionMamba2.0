# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
'Tensor utils.'
import inspect
import json
import math
import random
from collections.abc import Mapping
from collections.abc import Sequence
from functools import wraps
from math import prod
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Mapping
from typing import Optional
from typing import ParamSpec
from typing import TypeVar
from typing import Union
import numpy as np
import torch

def validate(validator, save_args: bool=False, super_init: bool=False):

    def decorator(func):

        @wraps(func)
        def validated_func(*args, **kwargs):
            conf = validator(**kwargs)
            if save_args:
                assert len(args) != 0
                obj = args[0]
                if super_init:
                    super(type(obj), obj).__init__()
                for (key, val) in conf.__dict__.items():
                    setattr(obj, key, val)
            return func(*args, conf)
        return validated_func
    return decorator
Tensor = Any
P = ParamSpec('P')
R = TypeVar('R')

def ensure_batched(**spec: int) -> Callable[[Callable[P, R]], Callable[P, R]]:
    if not spec:
        raise ValueError('At least one argument spec must be provided.')
    (canonical_name, canonical_ndim) = next(iter(spec.items()))

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        sig = inspect.signature(fn)

        @wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            if canonical_name not in bound.arguments:
                raise TypeError(f"Missing canonical argument '{canonical_name}'.")
            x0 = bound.arguments[canonical_name]
            if x0 is None:
                raise ValueError(f"Canonical '{canonical_name}' cannot be None.")
            expected_tail_dims = canonical_ndim - 1
            if x0.ndim < expected_tail_dims:
                raise ValueError(f"'{canonical_name}' ndim={x0.ndim} < expected {expected_tail_dims} tail dims.")
            if expected_tail_dims == 0:
                orig_batch_shape = x0.shape
                tail_shape = ()
            else:
                orig_batch_shape = x0.shape[:-expected_tail_dims]
                tail_shape = x0.shape[-expected_tail_dims:]
            B_flat = prod(orig_batch_shape) if orig_batch_shape else 1
            is_unbatched_input = len(orig_batch_shape) == 0
            is_already_flat = len(orig_batch_shape) == 1
            if is_unbatched_input:
                x0_batched = x0.reshape(1, *tail_shape)
            elif is_already_flat:
                x0_batched = x0
            else:
                x0_batched = x0.reshape(B_flat, *tail_shape)
            bound.arguments[canonical_name] = x0_batched
            for (name, target_ndim) in list(spec.items())[1:]:
                val = bound.arguments.get(name, None)
                if val is None:
                    continue
                arg_tail_dims = target_ndim - 1
                if val.ndim < arg_tail_dims:
                    raise ValueError(f"'{name}' ndim={val.ndim} too small.")
                if arg_tail_dims == 0:
                    val_batch_shape = val.shape
                    val_tail_shape = ()
                else:
                    val_batch_shape = val.shape[:-arg_tail_dims]
                    val_tail_shape = val.shape[-arg_tail_dims:]
                if len(val_batch_shape) == 0:
                    if not is_unbatched_input:
                        raise ValueError(f"'{name}' is unbatched but canonical is batched.")
                    val_batched = val.reshape(1, *val_tail_shape)
                else:
                    if val_batch_shape != orig_batch_shape:
                        raise ValueError(f"Batch dimensions mismatch! '{canonical_name}' has {orig_batch_shape}, but '{name}' has {val_batch_shape}.")
                    if is_already_flat:
                        val_batched = val
                    else:
                        val_batched = val.reshape(B_flat, *val_tail_shape)
                bound.arguments[name] = val_batched
            out = fn(**bound.arguments)

            def restore(obj):
                if isinstance(obj, Mapping):
                    return {k: restore(v) for (k, v) in obj.items()}
                if isinstance(obj, (list, tuple)):
                    return type(obj)((restore(x) for x in obj))
                if hasattr(obj, 'shape'):
                    if obj.ndim == 0:
                        return obj
                    if obj.shape[0] != B_flat:
                        return obj
                    if is_already_flat:
                        return obj
                    rest = obj.shape[1:]
                    if is_unbatched_input:
                        return obj.reshape(*rest)
                    return obj.reshape(*orig_batch_shape, *rest)
                return obj
            return restore(out)
        return wrapper
    return decorator

def to_numpy(obj):
    if isinstance(obj, Mapping):
        return {k: to_numpy(v) for (k, v) in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)((to_numpy(x) for x in obj))
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy()
    return obj

def to_torch(obj, device=None, dtype=None):
    if isinstance(obj, Mapping):
        return {k: to_torch(v, device, dtype) for (k, v) in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)((to_torch(x, device, dtype) for x in obj))
    if isinstance(obj, np.ndarray):
        obj = torch.from_numpy(obj)
    if isinstance(obj, torch.Tensor):
        if dtype is not None:
            obj = obj.to(dtype=dtype)
        if device is None:
            return obj
        return obj.to(device)
    return obj

def seed_everything(seed: int, deterministic: bool=False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_json(path: Union[str, Path]) -> Any:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'JSON file not found: {path}')
    try:
        with path.open('r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f'Invalid JSON in file {path}: {e}') from e

def save_json(path: Union[str, Path], data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open('w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except (TypeError, ValueError) as e:
        raise ValueError(f'Data is not JSON serializable: {e}') from e
