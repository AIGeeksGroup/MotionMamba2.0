"""Namespace conversion for explicitly supplied training metadata."""
from motion_mamba2.model.configuration import read_config
def target_namespace(value):
    if isinstance(value,dict):return {k:target_namespace(v) for k,v in value.items()}
    if isinstance(value,list):return [target_namespace(v) for v in value]
    if isinstance(value,str):
        if value.startswith('ardy.'):
            raise ValueError('Use the public class targets in configs/model.yaml')
    return value
def checkpoint_metadata(value):
    if not isinstance(value,dict):raise TypeError('Expected checkpoint metadata mapping')
    return value
