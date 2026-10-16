"""Explicit task identities and synchronous gradients for matched long FT."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def module_hash(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_plan(path, manifest, world, batch):
    plan = json.loads(Path(path).read_text())
    if plan['manifest_sha256'] != file_hash(manifest):
        raise ValueError('sample plan manifest changed')
    if world * batch != 4 or plan['global_batch'] != 4:
        raise ValueError('matched training requires effective global batch four')
    return plan


def sync_gradients(parameters):
    """Average the entire replay+loss graph, including dynamically used paths.

    Accepted history is built outside the ordinary denoiser forward. An
    explicit reduction after backward avoids DDP ready hooks on that graph.
    A common used mask ensures identical collectives on every rank.
    """
    used = torch.tensor([p.grad is not None for p in parameters],
                        device=parameters[0].device, dtype=torch.int32)
    dist.all_reduce(used, op=dist.ReduceOp.MAX)
    active = []
    for p, flag in zip(parameters, used.cpu().tolist()):
        if flag:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            active.append(p.grad)
    bucket, size = [], 0
    for gradient in active + [None]:
        if bucket and (gradient is None or size >= 8_000_000):
            flat = torch._utils._flatten_dense_tensors(bucket)
            dist.all_reduce(flat)
            flat.div_(dist.get_world_size())
            for target, value in zip(bucket, torch._utils._unflatten_dense_tensors(flat, bucket)):
                target.copy_(value)
            bucket, size = [], 0
        if gradient is not None:
            bucket.append(gradient)
            size += gradient.numel()


def validate_contract(args, world):
    expected = dict(model='g1', frames=312, sequential_rollout_windows=6,
                    sequential_rollout_min_windows=6, learning_rate=1e-4,
                    seed=9032026, history_probability=0.0,
                    sequential_cache_probability=1.0,
                    generated_history_probability_start=1.0,
                    generated_history_probability_1k=1.0,
                    generated_history_probability_5k=1.0,
                    generated_history_diffusion_steps=10,
                    generated_history_max_supervision_points=1,
                    generated_history_bptt_windows=4,
                    generated_history_attach_seed_write=True,
                    sequential_rollout_curriculum='d2_post_eviction_v1',
                    history_sampling_strategy='legacy_conditional',
                    constraint_dropout=0.1, text_dropout=0.1,
                    boundary_loss_weight=10.0, root_velocity_loss_weight=50.0,
                    constraint_loss_weight=1.0, fk_constraint_loss_weight=1.0,
                    foot_skate_loss_weight=50.0)
    bad = {k: (getattr(args, k), v) for k, v in expected.items() if getattr(args, k) != v}
    if bad or world * args.batch_size != 4 or not args.sample_plan:
        raise ValueError(f'matched_longft5k_v1 contract violation: {bad}')
    if args.steps != 5000 and not args.nonproduction_test:
        raise ValueError('formal matched stage requires exactly 5000 updates')
    if args.train_scope not in {'denoiser', 'state_memory'}:
        raise ValueError('unsupported matched parameter scope')
    if args.token_state_interaction != 'naive_gated_add':
        raise ValueError('Main3 excludes state_query and KV modulation')
    if args.train_scope == 'state_memory' and not args.content_memory_enabled:
        raise ValueError('added-module scope requires persistent memory')


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state())


def restore_rng(state):
    if 'python' in state:
        random.setstate(state['python'])
        np.random.set_state(state['numpy'])
