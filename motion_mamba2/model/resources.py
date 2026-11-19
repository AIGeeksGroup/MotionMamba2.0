"""Explicit resource bindings for unchanged training entry points."""
from functools import lru_cache
from pathlib import Path
import json,os
ROOT=Path(__file__).resolve().parents[2]
def workspace():return Path(os.environ.get('MM2_WORKSPACE','./workspace')).expanduser().resolve()
def repository_resource(relative):
    path=Path(relative)
    if path.is_absolute() or '..' in path.parts:raise ValueError(relative)
    target=ROOT/path
    if not target.exists():raise FileNotFoundError(target)
    return target
def artifact_path(key):
    filename=os.environ.get('MM2_ARTIFACT_MAP')
    bindings=json.loads(Path(filename).read_text()) if filename else {}
    value=bindings.get(key,{})
    value=value.get('path') if isinstance(value,dict) else value
    if not value:raise ValueError('Set resource path for '+key+' in MM2_ARTIFACT_MAP')
    return os.path.expandvars(os.path.expanduser(value))
def resolve_tree(value):
    if isinstance(value,dict):return {k:resolve_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [resolve_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(resolve_tree(v) for v in value)
    if isinstance(value,str) and value.startswith('artifact://'):return artifact_path(value[11:])
    return value
