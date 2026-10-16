"""Incremental G1 rollout using the audited ARDY post-recenter commit path.

Model weights are shared; every session owns KV/SSM/content memory, world
translation, heading and CUDA RNG. Sessions are serialized by the service.
Only unpublished window boundaries may be rewound.
"""
from __future__ import annotations

import copy
import os
import time
from dataclasses import dataclass, field, fields, is_dataclass
from collections import OrderedDict

import numpy as np
import torch

from ardy.model.ardy_model import translate_normalized_root_motion
from ardy.exports.mujoco import MujocoQposConverter
from g1_online_controls import conditions, control_fingerprint
from g1_control_intents import compile_intent


def copy_state(value, device):
    """Copy explicit state containers, including frozen cache dataclasses."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device, copy=True)
    if isinstance(value, torch.device):
        return torch.device(device)
    if is_dataclass(value):
        return type(value)(**{f.name: copy_state(getattr(value, f.name), device) for f in fields(value)})
    if isinstance(value, list):
        return [copy_state(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(copy_state(v, device) for v in value)
    if isinstance(value, dict):
        return {k: copy_state(v, device) for k, v in value.items()}
    if hasattr(value, '__dict__'):
        result = copy.copy(value)
        for key, v in vars(value).items():
            setattr(result, key, copy_state(v, device))
        return result
    return value


@dataclass
class Session:
    seed: int
    total_windows: int
    snapshots: OrderedDict = field(default_factory=OrderedDict)
    windows: int = 0
    published: int = 0
    revision: int = 0
    last_result: dict | None = None
    last_prompt: str | None = None
    last_request: tuple | None = None


class OnlineG1:
    def __init__(self, model, client, retained_boundaries=4):
        self.model, self.client = model, client
        self.controller = client.controller
        self.converter = MujocoQposConverter(model.skeleton)
        self.retained_boundaries = retained_boundaries
        self.steps = 10
        # Debug capture is opt-in for isolated decoder-contract audits.  It is
        # intentionally disabled for the production service because these
        # tensors are sizeable and are not part of the public API contract.
        self.capture_runtime_latents = os.environ.get('G1_CAPTURE_RUNTIME_LATENTS', '0') == '1'

    def new_session(self, seed, total_windows=0, translation=None, heading=0.0):
        # Zero explicitly means continuous; finite sessions remain supported.
        if not 0<=total_windows<=300:
            raise ValueError('total_windows must be 0 (continuous) or 1–300')
        translation = torch.as_tensor(translation if translation is not None else [0, 0, 0], dtype=torch.float32).reshape(1, 3)
        heading = torch.as_tensor([heading], dtype=torch.float32)
        if not torch.isfinite(translation).all() or not torch.isfinite(heading).all():
            raise ValueError('initial transform must be finite')
        self.client.reset_memory()
        rng = torch.Generator(device=self.model.device).manual_seed(seed).get_state()
        s = Session(seed=int(seed),total_windows=total_windows)
        s.snapshots[0] = dict(root=None, body=None, translation=translation,
                              heading=heading, rng=rng.cpu().clone(), windows=0,
                              root_commits=0, body_commits=0, next_chunk_id=0)
        return s

    def restore(self, snapshot):
        c = self.controller
        c.reset()
        c.root_block._sequence_state = copy_state(snapshot['root'], self.model.device)
        c.body_block._sequence_state = copy_state(snapshot['body'], self.model.device)
        c.root_block.accepted_commits = snapshot['root_commits']
        c.body_block.accepted_commits = snapshot['body_commits']
        c._completed_windows = c._accepted_commits = snapshot['windows']
        # Real-prefix snapshots may already contain an accepted history commit
        # while their generated-window index is still zero.  Older snapshots
        # did not persist this field, so retain the historical fallback.
        c._next_chunk_id = snapshot.get('next_chunk_id', snapshot['windows'])

    def rewind(self, s, boundary, revision):
        if revision != s.revision:
            raise ValueError('stale session revision')
        if boundary < s.published:
            raise ValueError('cannot rewrite motion already published to SONIC')
        if boundary not in s.snapshots:
            raise ValueError('boundary outside retained snapshot range')
        for n in list(s.snapshots):
            if n > boundary:
                del s.snapshots[n]
        s.windows = boundary
        s.revision += 1
        s.last_result = None
        s.last_prompt = None
        s.last_request = None
        return dict(windows=s.windows, published=s.published, revision=s.revision)

    def publish(self, s, boundary, revision):
        if revision != s.revision or not s.published <= boundary <= s.windows:
            raise ValueError('invalid publication boundary or stale revision')
        s.published = boundary
        for n in list(s.snapshots):
            if n < boundary:
                del s.snapshots[n]

    @torch.inference_mode()
    def step(self, s, text, text_mask, prompt, expected_window, revision, control=None, request_version=0):
        if revision != s.revision:
            raise ValueError('stale session revision')
        request = (prompt, int(request_version), control_fingerprint(control))
        if expected_window == s.windows-1 and s.last_result is not None and request == s.last_request:
            return s.last_result  # retried HTTP request cannot commit twice
        if expected_window != s.windows:
            raise ValueError('unexpected window index')
        if s.total_windows and s.windows>=s.total_windows:
            raise ValueError('session horizon exhausted')
        if s.windows-s.published >= self.retained_boundaries:
            raise ValueError('publish or rewind buffered windows before generating more')
        started = time.perf_counter()
        before = s.snapshots[s.windows]
        m, c = self.model, self.controller
        device = m.device
        self.restore(before)
        heading = before['heading'].to(device)
        translation = before['translation'].to(device)
        nsteps = torch.tensor([self.steps], device=device)
        m.diffusion.calc_diffusion_vars(m.diffusion.space_timesteps(nsteps[0])[0])
        # KV/SSM carries global chunk IDs. The denoiser's temporal masks and
        # spatial conditions are local to this window, never session length.
        control_origin=before.get('control_origin', translation).to(device)
        effective_control,intent_state,intent_receipt=compile_intent(
            control,before.get('intent_state'),control_origin[0,[0,2]].cpu().tolist(),
            before.get('root_velocity_xz',[0.,0.]),float(heading[0]),s.windows*52)
        zero, mask = conditions(m, effective_control, control_origin)
        # Coordinate-local history is absent: accepted history is in KV/SSM/PCM.
        # This is crop_history_length=0 from the validated full rollout.
        try:
            with torch.random.fork_rng(devices=[torch.device(device).index or 0]):
                torch.cuda.set_rng_state(before['rng'], device)
                sequence = m._generate_window(
                    None, translation, 0, 0, 52, text, text_mask, heading,
                    mask, zero, nsteps, (2.0,2.0), list(range(self.steps))[::-1],
                    progress_bar=lambda x:x, target_motion=None, cfg_type=None)
                sequence, center, root = m._recenter_history(sequence, torch.tensor([51],device=device), requantize=True)
                translation = translation + center
                heading = m.motion_rep.get_root_heading_angle(m.motion_rep.global_root_stats.unnormalize(root))[:,-1]
                m._sequence_kernel_post_window_hook(sequence, center, root, heading, s.windows)
                root, body = m.hybrid.get_root_and_latent_body_motion_from_hybrid(sequence)
                root = translate_normalized_root_motion(root, translation, m.motion_rep)
                world = m.hybrid.get_hybrid_motion_from_root_and_latent_body_motion(root, body)
                decoder_context = before.get('decoder_history')
                decoder_context_before = None if decoder_context is None else decoder_context.detach().clone()
                if getattr(m, 'decoder_history_frames', None) is not None:
                    from g1_motion_contract import decode_committed_window
                    explicit, decoder_context = decode_committed_window(m, world, decoder_context)
                else:
                    explicit = m.hybrid.get_explicit_motion_from_hybrid_autoregressive(
                        world, torch.ones(1,52,device=device,dtype=torch.bool),
                        torch.tensor([52],device=device), motion_mask=mask[:,:52], crop_history_length=0)
                decoded = m.motion_rep.inverse(explicit, is_normalized=True)
                qpos = np.asarray(self.converter.dict_to_qpos(decoded,device=device,numpy=True)).reshape(52,36)
                if not np.isfinite(qpos).all():
                    raise ValueError('nonfinite generated motion')
                expected = s.windows+1
                if c._accepted_commits != expected or c.root_block.accepted_commits != before['root_commits']+1 or c.body_block.accepted_commits != before['body_commits']+1:
                    raise RuntimeError('window must commit root/body memory exactly once')
                snapshot = dict(root=copy_state(c.root_block._sequence_state,'cpu'),
                                body=copy_state(c.body_block._sequence_state,'cpu'),
                                translation=translation.cpu().clone(), heading=heading.cpu().clone(),
                                rng=torch.cuda.get_rng_state(device).cpu().clone(), windows=expected,
                                root_commits=c.root_block.accepted_commits,body_commits=c.body_block.accepted_commits,
                                next_chunk_id=c._next_chunk_id)
                snapshot['control_origin'] = translation.cpu().clone()
                physical=m.motion_rep.unnormalize(explicit)
                snapshot['root_velocity_xz']=((physical[0,-1,[0,2]]-physical[0,-6,[0,2]])*5).cpu().tolist()
                snapshot['intent_state']=intent_state
                if getattr(m, 'decoder_history_frames', None) is not None:
                    snapshot['decoder_history'] = decoder_context.cpu().clone()
                torch.cuda.synchronize()
                result = dict(window=s.windows, revision=s.revision, frames=52, fps=25,
                          normalized_motion=explicit[0].float().cpu().tolist(),
                          prompt=prompt, qpos=qpos.tolist(), generation_seconds=time.perf_counter()-started,
                          accepted_commits=expected,
                          request_version=int(request_version), control_sha256=request[2],
                          condition_nonzero=int(mask.sum().item()), condition_shape=list(mask.shape),
                          source_start=s.windows*52, generated_at=time.time(),
                          cache_chunk_ids=list(c.root_block._sequence_state.layers[0].attention_cache.chunk_ids))
                result.update(effective_control=effective_control,intent_receipt=intent_receipt)
            if self.capture_runtime_latents:
                # These are the exact tensors consumed by the committed decoder
                # for this accepted window, rather than a re-encoding of the
                # serialized explicit motion.  Keep them outside the normal
                # response unless the isolated audit explicitly enables capture.
                result.update(
                    runtime_latent_contract='g1_runtime_world_v1',
                    world_hybrid=world[0].float().cpu().tolist(),
                    decoder_context_before=(None if decoder_context_before is None
                                             else decoder_context_before[0].float().cpu().tolist()),
                    decoder_context_after=(None if decoder_context is None
                                            else decoder_context[0].float().cpu().tolist()),
                    decoder_context_tokens=(0 if decoder_context is None else int(decoder_context.shape[1])),
                )
            s.snapshots[expected] = snapshot
            s.windows = expected
            s.last_prompt, s.last_result = prompt, result
            s.last_request = request
            return result
        except Exception:
            self.restore(before)
            raise
