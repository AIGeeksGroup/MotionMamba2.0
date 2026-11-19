'Training launch.'
import importlib
from motion_mamba2.model.kernel_launch import install_forward_launch_contract
from motion_mamba2.model.kernel_launch import FORWARD_LAUNCH_CONTRACT
TRAINING_LAUNCH_CONTRACT = 'mamba3_backward_w4s1_dz32_dt64_v1'
SPECS = [('motion_mamba2.kernels.angle_step', 'angle_dt_bwd_kernel', {}), ('motion_mamba2.kernels.mamba_backward', 'mamba3_siso_bwd_kernel_dzdo', {'CHUNK_SIZE': 32}), ('motion_mamba2.kernels.mamba_backward', 'mamba3_siso_bwd_kernel_dqkv', {}), ('motion_mamba2.kernels.mamba_backward', 'mamba3_siso_bwd_kernel_rotary_bias_angles', {}), ('motion_mamba2.kernels.mamba_backward', 'mamba3_siso_bwd_kernel_ddt_dtrap_dinput_states', {'CHUNK_SIZE': 64})]

def install_training_launch_contract(version=TRAINING_LAUNCH_CONTRACT):
    if version != TRAINING_LAUNCH_CONTRACT:
        raise ValueError(f'Unknown training launch contract: {version}')
    install_forward_launch_contract(FORWARD_LAUNCH_CONTRACT)
    selected = []
    for (module, name, kwargs) in SPECS:
        kernel = getattr(importlib.import_module(module), name)
        configs = [c for c in kernel.configs if c.num_warps == 4 and c.num_stages == 1 and (c.kwargs == kwargs) and (getattr(c, 'num_ctas', 1) == 1) and (getattr(c, 'maxnreg', None) is None)]
        if len(configs) != 1:
            raise RuntimeError(f'Registered launch configuration unavailable: {name}')
        if kernel.cache:
            raise RuntimeError(f'Install contract before first backward: {name}')
        selected.append((module, name, kernel, configs[0]))
    for (_, _, kernel, config) in selected:
        kernel.configs = [config]
    return dict(version=version, kernels={module + '.' + name: str(config) for (module, name, _, config) in selected})
