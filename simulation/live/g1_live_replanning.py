"""Transactional window replay with a fixed, token-aligned consumed prefix.

Wrap the loaded (possibly frozen) runtime; do not change training or its codec.
The ordinary generation path remains identical. Replacement candidates do not
change the live session until the bridge verifies its consumption boundary.
"""
import copy
from collections import OrderedDict
from pathlib import Path
import tempfile
import uuid

import numpy as np
import torch
from ardy.model.ardy_model import translate_normalized_root_motion
import g1_online_runtime as runtime_module
OriginalOnlineG1 = getattr(runtime_module, 'OriginalOnlineG1', runtime_module.OnlineG1)


class ReplanningOnlineG1(OriginalOnlineG1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.archive_root = Path(tempfile.mkdtemp(prefix='mm2-live-history-'))

    def new_session(self, *args, **kwargs):
        s = super().new_session(*args, **kwargs)
        s.archive = self.archive_root / uuid.uuid4().hex
        s.archive.mkdir()
        s.boundaries = {}
        s.records = {}
        s.pending_replan = None
        self._save_boundary(s, 0)
        return s

    def _save_boundary(self, s, boundary):
        path = s.archive / f'r{s.revision}_b{boundary}.pt'
        torch.save(s.snapshots[boundary], path)
        s.boundaries[boundary] = path

    def _boundary(self, s, boundary):
        if boundary in s.snapshots:
            return s.snapshots[boundary]
        if boundary not in s.boundaries:
            raise ValueError('no accepted history at replacement boundary')
        return torch.load(s.boundaries[boundary], map_location='cpu', weights_only=False)

    def step(self, s, *args, **kwargs):
        previous = s.windows
        capture = self.capture_runtime_latents
        self.capture_runtime_latents = True
        try:
            value = super().step(s, *args, **kwargs)
        finally:
            self.capture_runtime_latents = capture
        if s.windows != previous:
            record = dict(world=torch.tensor(value['world_hybrid'], dtype=torch.float32)[None],
                          normalized_motion=value['normalized_motion'], qpos=value['qpos'])
            path = s.archive / f'r{s.revision}_w{previous}.pt'
            torch.save(record, path)
            s.records[previous] = path
            self._save_boundary(s, s.windows)
            for key in ('runtime_latent_contract', 'world_hybrid', 'decoder_context_before',
                        'decoder_context_after', 'decoder_context_tokens'):
                value.pop(key, None)
        return value

    @torch.inference_mode()
    def prepare_replan(self, s, text, text_mask, prompt, source_frame, revision,
                       control=None, request_version=0, lookahead_windows=0,
                       lookahead_control=None):
        if revision != s.revision:
            raise ValueError('stale session revision')
        if lookahead_windows not in (0, 1):
            raise ValueError('replacement lookahead must be zero or one window')
        frame = int(source_frame)
        if frame < 0 or frame % 4 or frame >= s.windows * 52:
            raise ValueError('replacement must be a generated, four-frame boundary')
        boundary, prefix = divmod(frame, 52)
        before = self._boundary(s, boundary)
        old = torch.load(s.records[boundary], map_location='cpu', weights_only=False)
        if prefix and (control or {}).get('version') == 'g1_joint_interaction_v1':
            # A newly changed keyboard intent starts at the accepted cut point,
            # rather than spending its ramp inside the immutable prefix.
            before = runtime_module.copy_state(before, 'cpu')
            physical = self.model.motion_rep.unnormalize(
                torch.tensor(old['normalized_motion'], device=self.model.device)[None])
            before['control_origin'] = physical[:,prefix-1,:3].cpu().clone()
            previous = max(0,prefix-6)
            before['root_velocity_xz'] = ((physical[0,prefix-1,[0,2]]-
                physical[0,previous,[0,2]])*25/max(prefix-1-previous,1)).cpu().tolist()
            before['control_heading'] = float(self.model.motion_rep.get_root_heading_angle(
                physical[:,prefix-1:prefix])[0,0])
            control = dict(control, transition_source_frame=frame)
        trial = copy.copy(s)
        trial.snapshots = OrderedDict([(boundary, before)])
        trial.boundaries = {k: v for k, v in s.boundaries.items() if k <= boundary}
        trial.records = {k: v for k, v in s.records.items() if k < boundary}
        trial.windows = trial.published = boundary
        trial.revision = s.revision + 1
        trial.last_request = trial.last_result = trial.last_prompt = None
        trial.pending_replan = None
        m = self.model
        original = m.denoising_step
        clean = old['world'].to(m.device)
        root, body = m.hybrid.get_root_and_latent_body_motion_from_hybrid(clean)
        root = translate_normalized_root_motion(root, -before['translation'].to(m.device), m.motion_rep)
        clean = m.hybrid.get_hybrid_motion_from_root_and_latent_body_motion(root, body)
        tokens = prefix // 4
        noise = None

        def fixed_prefix(x, *step_args, **step_kwargs):
            nonlocal noise
            t = step_args[11]  # ArdyModel.denoising_step: timestep after text masks.
            if tokens:
                if noise is None:
                    noise = x[:, :tokens].clone()
                x = x.clone()
                x[:, :tokens] = m.diffusion.q_sample(clean[:, :tokens], t, noise=noise)
            result = original(x, *step_args, **step_kwargs)
            if tokens:
                result = result.clone()
                if int(t[0]) == 0:
                    result[:, :tokens] = clean[:, :tokens]
                else:
                    result[:, :tokens] = m.diffusion.q_sample(clean[:, :tokens], t - 1, noise=noise)
            return result

        # Serialized by the service lock; always restore even after an error.
        m.denoising_step = fixed_prefix
        try:
            value = self.step(trial, text, text_mask, prompt, boundary, trial.revision,
                              control, request_version)
        finally:
            m.denoising_step = original
        # The codec can have temporal receptive fields. Preserve the actual
        # accepted explicit prefix as well, instead of merely trusting latents.
        value['normalized_motion'][:prefix] = old['normalized_motion'][:prefix]
        explicit = torch.tensor(value['normalized_motion'], device=m.device)[None]
        decoded = m.motion_rep.inverse(explicit, is_normalized=True)
        value['qpos'] = np.asarray(self.converter.dict_to_qpos(decoded, device=m.device,
                                  numpy=True)).reshape(52, 36).tolist()
        value['qpos'][:prefix] = old['qpos'][:prefix]
        saved = torch.load(trial.records[boundary], weights_only=False)
        saved.update(normalized_motion=value['normalized_motion'], qpos=value['qpos'])
        torch.save(saved, trial.records[boundary])
        following = []
        if lookahead_control is None:
            # Sparse points belong to their original window; do not silently
            # shift them forward by 52 frames when preparing a guard window.
            lookahead_control = {k:v for k,v in (control or {}).items() if k != 'root2d'}
        for _ in range(lookahead_windows):
            following.append(self.step(trial, text, text_mask, prompt, trial.windows,
                                        trial.revision, lookahead_control, request_version))
        token = uuid.uuid4().hex
        value.update(replan_token=token, replacement_source_frame=frame,
                     frozen_prefix_frames=prefix, base_revision=s.revision,
                     following_windows=following)
        s.pending_replan = (token, s.revision, trial)
        return value

    def accept_replan(self, s, token, revision):
        if getattr(s, 'last_accepted_replan', None) == (token, revision) and s.revision == revision + 1:
            return dict(windows=s.windows, published=s.published, revision=s.revision)
        pending = s.pending_replan
        if pending is None or pending[0] != token or pending[1] != revision or s.revision != revision:
            raise ValueError('stale or missing replacement candidate')
        trial = pending[2]
        s.__dict__.update(trial.__dict__)
        s.pending_replan = None
        s.last_accepted_replan = (token, revision)
        return dict(windows=s.windows, published=s.published, revision=s.revision)

    def cancel_replan(self, s):
        s.pending_replan = None
        return dict(windows=s.windows, published=s.published, revision=s.revision)
