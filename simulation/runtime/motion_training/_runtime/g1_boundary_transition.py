"""Opt-in *label construction*, never an inference postprocessor.

An inertial transition preserves the accepted boundary pose and first
derivative, and removes the residual relative to a natural donor in 12 frames.
This is a proposal, not a certificate: raw and actual-codec checks are required.
"""
import numpy as np
import torch
from scipy.spatial.transform import Rotation


def inertial_transition(rep, accepted_tail, donor_with_boundary, frames=12):
    if accepted_tail.shape[1] != 2 or donor_with_boundary.shape[1] != 54:
        raise ValueError('Expected two accepted frames and two donor history + 52 future frames')
    if frames < 4 or frames > 26:
        raise ValueError('Transition must occupy 4..26 frames, not the full target')
    if len(accepted_tail) != 1 or not all(torch.isfinite(v).all() for v in (accepted_tail, donor_with_boundary)):
        raise ValueError('Single finite sample required')
    a = rep.inverse(accepted_tail, is_normalized=False)
    d = rep.inverse(donor_with_boundary, is_normalized=False)
    ar = a['local_rot_mats'].detach().cpu().double().numpy()[0]
    dr = d['local_rot_mats'].detach().cpu().double().numpy()[0]
    n = ar.shape[1]
    delta = Rotation.from_matrix(ar[-1] @ dr[1].transpose(0, 2, 1)).as_rotvec()
    av = Rotation.from_matrix(ar[-1] @ ar[-2].transpose(0, 2, 1)).as_rotvec()
    dv = Rotation.from_matrix(dr[1] @ dr[0].transpose(0, 2, 1)).as_rotvec()
    u = np.minimum(np.arange(1, 53) / frames, 1.)
    h0 = 2*u**3 - 3*u**2 + 1
    h1 = u**3 - 2*u**2 + u
    correction = h0[:, None, None]*delta + h1[:, None, None]*frames*(av-dv)
    rotations = Rotation.from_rotvec(correction.reshape(-1, 3)).as_matrix().reshape(52, n, 3, 3) @ dr[2:]
    rotations = torch.as_tensor(rotations, device=accepted_tail.device, dtype=accepted_tail.dtype)[None]
    h0 = torch.as_tensor(h0, device=accepted_tail.device, dtype=accepted_tail.dtype)[None, :, None]
    h1 = torch.as_tensor(h1, device=accepted_tail.device, dtype=accepted_tail.dtype)[None, :, None]
    p, q = a['root_positions'], d['root_positions']
    position_delta = p[:, -1:] - q[:, 1:2]
    velocity_delta = (p[:, -1:] - p[:, -2:-1]) - (q[:, 1:2] - q[:, :1])
    root = q[:, 2:] + h0*position_delta + h1*frames*velocity_delta
    # Compute coherent FK, velocities and contact features. Accepted history
    # is used for derivatives but is never returned/replaced in the rollout.
    merged = rep(torch.cat((a['local_rot_mats'], rotations), 1),
                 torch.cat((p, root), 1), to_normalize=False)
    return merged[:, 2:]


def encode_proposed_target(model, actual_decoded_history, target, actual_tokens):
    """Encode the proposed future, decode with immutable ACTUAL prefix tokens."""
    from g1_motion_contract import decode_committed_window
    context = torch.cat((actual_decoded_history[:, -52:], model.motion_rep.normalize(target)), 1)
    pad = torch.ones(context.shape[:2], device=context.device, dtype=torch.bool)
    tokens, _ = model.hybrid.get_hybrid_motion_from_explicit(context, pad.sum(1), pad)
    tokens = tokens[:, -13:]
    decoded, _ = decode_committed_window(model, tokens, actual_tokens)
    return tokens, decoded
