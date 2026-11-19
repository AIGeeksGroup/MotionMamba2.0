"""Tensor-only checkpoint and explicit architecture construction helpers."""
from pathlib import Path
import torch

def instantiate_from_dict(cfg,overrides=None):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    value=dict(cfg)
    if overrides:value.update(overrides)
    return instantiate(OmegaConf.create(value))

def load_checkpoint_state_dict(path):
    path=Path(path)
    if path.suffix=='.safetensors':
        from safetensors.torch import load_file
        return load_file(str(path),device='cpu')
    value=torch.load(path,map_location='cpu',weights_only=True)
    if isinstance(value,dict) and 'state_dict' in value:value=value['state_dict']
    if not isinstance(value,dict) or not all(isinstance(x,torch.Tensor) for x in value.values()):
        raise ValueError('Expected a tensor state dictionary')
    return value

def get_env_var(name,default=None):
    import os
    return os.environ.get(name) or default
