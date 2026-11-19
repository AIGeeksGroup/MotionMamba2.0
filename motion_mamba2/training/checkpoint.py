"""Atomic full-state checkpoints and deterministic state fingerprints."""
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def tensor_hash(value):
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(str((item.dtype, item.shape)).encode())
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode())
            for part in item:
                visit(part)
        else:
            digest.update(repr(item).encode())

    visit(value)
    return digest.hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state())


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state(state['cuda'])


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def save(path, raw, optimizer, completed, contract, parallel):
    # Collect rank RNGs before serializing. All model/optimizer states must agree.
    hashes = parallel.gather(tensor_hash([raw.state_dict(), optimizer.state_dict()]))
    if len(set(hashes)) != 1:
        raise RuntimeError('Ranks disagree at checkpoint boundary')
    rngs = parallel.gather(rng_state())
    if parallel.rank() == 0:
        path = Path(path)
        temporary = path.with_suffix('.tmp')
        if path.exists():
            raise FileExistsError(f'Checkpoint already exists: {path}')
        payload = dict(format='motion_mamba2_training', model=raw.state_dict(),
                       optimizer=optimizer.state_dict(), completed=completed,
                       scheduler=dict(completed=completed), sampler_cursor=completed * 512,
                       rng_by_rank=rngs, world_size=parallel.world(), contract=contract,
                       state_hash=hashes[0])
        with temporary.open('wb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        atomic_json(path.parent / 'latest.json', dict(step=completed, checkpoint=path.name,
                    sha256=file_hash(path), state_hash=hashes[0], sampler_cursor=completed * 512))
    parallel.barrier()


def resume(path, raw, optimizer, contract, parallel):
    # Full training checkpoints are trusted local outputs, not public downloads.
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('format') != 'motion_mamba2_training' or payload['contract'] != contract:
        raise ValueError('Checkpoint protocol, source or resources differ')
    if payload['world_size'] != parallel.world() or len(payload['rng_by_rank']) != parallel.world():
        raise ValueError('Resume with the same GPU count')
    completed = payload['completed']
    if payload['sampler_cursor'] != completed * 512 or payload['scheduler']['completed'] != completed:
        raise ValueError('Invalid sampler or scheduler position')
    raw.load_state_dict(payload['model'], strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    if tensor_hash([raw.state_dict(), optimizer.state_dict()]) != payload['state_hash']:
        raise ValueError('Restored model/Adam/master state differs')
    restore_rng(payload['rng_by_rank'][parallel.rank()])
    return completed


def export_model(path, raw):
    """Export inference tensors only; omit optimizer and run metadata."""
    path = Path(path)
    state = raw.state_dict()
    weights = dict(root_model=raw.root_model.state_dict(), body_model=raw.body_model.state_dict(),
                   denoiser_aux={key: value for key, value in state.items()
                                  if not key.startswith(('root_model.', 'body_model.'))})
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        torch.save(weights, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
