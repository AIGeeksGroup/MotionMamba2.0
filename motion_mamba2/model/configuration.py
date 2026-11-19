"""Read portable configuration without private filesystem defaults."""
import os
from pathlib import Path
import shlex
import yaml

ROOT=Path(__file__).resolve().parents[2]

def load_environment(path=None):
    path=Path(path or ROOT/'configs/paths.env').expanduser().resolve()
    values={}
    for number,line in enumerate(path.read_text().splitlines(),1):
        line=line.strip()
        if not line or line.startswith('#'):continue
        if '=' not in line:raise ValueError(f'{path}:{number}: expected KEY=value')
        name,raw=line.split('=',1)
        parts=shlex.split(raw,comments=True)
        if len(parts)>1:raise ValueError(f'{path}:{number}: quote paths containing spaces')
        values[name.strip()]=parts[0] if parts else ''
    for key,value in values.items():
        value=os.environ.get(key) or value
        if value and (key.endswith(('_PATH','_DIR','_ROOT','_CHECKPOINT','_BASE','_ADAPTER','_MNTP'))):
            p=Path(value).expanduser()
            value=str((ROOT/p).resolve() if not p.is_absolute() else p)
        values[key]=value
        if value:os.environ[key]=value
    frontend=values.get('MM2_FRONTEND_DIR')
    if frontend:
        for key,relative in [('MM2_CODEC_CHECKPOINT','tokenizer.safetensors'),('MM2_NORMALIZATION_PATH','stats/motion')]:
            if not values.get(key):values[key]=str(Path(frontend)/relative);os.environ[key]=values[key]
    return values

def require_path(values,key,*,directory=False):
    if not values.get(key):raise ValueError(f'Fill {key} in configs/paths.env')
    p=Path(values[key]).expanduser()
    if not (p.is_dir() if directory else p.is_file()):raise FileNotFoundError(f'{key}: {p}')
    return p

def read_config(path):
    p=Path(path)
    if not p.is_absolute():p=ROOT/p
    value=yaml.safe_load(p.read_text())
    if not isinstance(value,dict):raise ValueError(f'Expected a configuration mapping: {p}')
    return value
