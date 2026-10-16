"""Explicit, process-local Mamba-3 forward launch settings for reproducible evals.

This does not enable deterministic training or alter weights. Backward kernels
are untouched. Call before the first Mamba forward in an isolated evaluator.
"""
import hashlib
import importlib
from pathlib import Path

FORWARD_LAUNCH_CONTRACT = 'mamba3_siso_angle_w4s1_v1'
_INSTALLED = None
KERNELS = {
    'siso_forward': ('mamba_ssm.ops.triton.mamba3.mamba3_siso_fwd', 'mamba3_siso_fwd_kernel'),
    'angle_forward': ('mamba_ssm.ops.triton.mamba3.angle_dt', 'angle_dt_fwd_kernel'),
}


def _select_registered_config(kernel):
    if not hasattr(kernel, 'configs') or not hasattr(kernel, 'cache'):
        raise TypeError('Unsupported Triton autotuner API')
    selected = [c for c in kernel.configs if c.num_warps == 4 and c.num_stages == 1
                and getattr(c, 'num_ctas', 1) == 1 and getattr(c, 'maxnreg', None) is None
                and not c.kwargs]
    if len(selected) != 1:
        raise ValueError('Expected exactly one registered w4/s1/default-register configuration')
    return selected[0]


def _kernels():
    result = {}
    for label, (path, name) in KERNELS.items():
        module = importlib.import_module(path)
        result[label] = (getattr(module, name), Path(module.__file__))
    return result


def install_forward_launch_contract(version):
    global _INSTALLED
    if version != FORWARD_LAUNCH_CONTRACT:
        raise ValueError(f'Unknown forward launch contract: {version}')
    kernels = _kernels()
    selected = {label: _select_registered_config(kernel) for label, (kernel, _) in kernels.items()}
    if _INSTALLED is not None:
        if _INSTALLED != version or any(len(kernel.configs) != 1 for kernel, _ in kernels.values()):
            raise RuntimeError('Forward launch contract changed after installation')
        return describe_forward_launch_contract()
    if any(kernel.cache for kernel, _ in kernels.values()):
        raise RuntimeError('Install forward launch contract before any autotuned forward')
    # Validate every kernel before mutating either one; do not edit dependencies.
    for label, (kernel, _) in kernels.items():
        kernel.configs = [selected[label]]
    _INSTALLED = version
    return describe_forward_launch_contract()


def describe_forward_launch_contract():
    return dict(version=_INSTALLED or 'legacy_autotune', kernels={
        label: dict(source=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    configurations=[str(c) for c in kernel.configs],
                    chosen=[dict(key=repr(k), configuration=str(v)) for k, v in kernel.cache.items()])
        for label, (kernel, path) in _kernels().items()},
        note='Forward launch settings only; not a quality gate or a deterministic-backward guarantee.')
