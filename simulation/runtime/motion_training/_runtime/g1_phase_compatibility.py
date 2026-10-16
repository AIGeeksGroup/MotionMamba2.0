"""Calibrated, rejectable continuation candidates (development, not policy).

No diffusion output is modified. Scales are fitted only to trusted continuous
GT. Callers must supply candidate permission from a separate stage annotation;
physical similarity alone is not evidence of semantic compatibility.
"""
import math

import torch

from g1_continuation_target import continuation_target


UNITS = dict(pose='m', joint_velocity='m/s', root_velocity='m/s',
             root_height='m', rotation='rad', contact='fraction')


def history_difference(a, b, *, fps):
    """Physical errors between two two-frame histories in the same heading.

    Position/velocity comes from FK poses, not redundant feature velocities.
    Translation is factored out of articulation, but root velocity/height are
    separately retained. Rotations use local joint geodesic angle, in radians.
    """
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid fps')
    for g in (a, b):
        p = g['posed_joints']
        if p.ndim != 4 or p.shape[1] != 2 or p.shape[-1] != 3:
            raise ValueError('Expected [B,2,J,3] FK poses')
        r = g['local_rot_mats']
        if r.shape != (*p.shape[:-1], 3, 3):
            raise ValueError('Rotations must match FK poses')
        if g['foot_contacts'].shape[:2] != p.shape[:2]:
            raise ValueError('Contacts must match history')
        if not all(torch.isfinite(g[k]).all() for k in
                   ('posed_joints', 'local_rot_mats', 'foot_contacts')):
            raise ValueError('Nonfinite history geometry')
    pa, pb = a['posed_joints'], b['posed_joints']
    if pa.shape != pb.shape or a['foot_contacts'].shape != b['foot_contacts'].shape:
        raise ValueError('History shapes differ')
    va, vb = (pa[:, 1]-pa[:, 0])*fps, (pb[:, 1]-pb[:, 0])*fps
    rela, relb = pa[:, -1]-pa[:, -1, :1], pb[:, -1]-pb[:, -1, :1]
    relative_rotation = a['local_rot_mats'][:, -1].transpose(-1, -2) @ b['local_rot_mats'][:, -1]
    # atan2 is stable both at identity and near pi; no clamped acos floor.
    skew = torch.stack((relative_rotation[..., 2, 1]-relative_rotation[..., 1, 2],
                        relative_rotation[..., 0, 2]-relative_rotation[..., 2, 0],
                        relative_rotation[..., 1, 0]-relative_rotation[..., 0, 1]), -1)
    sine = .5*skew.norm(dim=-1)
    cosine = .5*(relative_rotation.diagonal(dim1=-2, dim2=-1).sum(-1)-1)
    angle = torch.atan2(sine, cosine)
    return dict(pose=(rela[:, 1:]-relb[:, 1:]).square().sum(-1).mean(-1).sqrt(),
                joint_velocity=((va-va[:, :1])[:, 1:]-(vb-vb[:, :1])[:, 1:]).square().sum(-1).mean(-1).sqrt(),
                root_velocity=(va[:, 0]-vb[:, 0]).norm(dim=-1),
                root_height=(pa[:, -1, 0, 1]-pb[:, -1, 0, 1]).abs(),
                rotation=angle.square().mean(-1).sqrt(),
                contact=(a['foot_contacts'][:, -1].bool() ^ b['foot_contacts'][:, -1].bool()).float().mean(-1))


def fit_calibration(errors, *, fps, contact_channels):
    """q95 scales and q99 per-component bounds from adjacent GT histories.

    Floors only avoid degenerate constant dimensions; the contact floor is
    one label. These are declared development choices, never acceptance rules
    for final motion. A held-out GT retention audit is required by the caller.
    """
    if not math.isfinite(fps) or fps <= 0 or contact_channels < 1:
        raise ValueError('Invalid calibration contract')
    floors = {k: (1/contact_channels if k == 'contact' else 1e-6) for k in UNITS}
    components = {}
    sizes = set()
    for key in UNITS:
        values = errors[key].detach().double().flatten()
        if len(values) < 32 or not torch.isfinite(values).all() or (values < 0).any():
            raise ValueError('Need at least 32 finite nonnegative GT observations')
        sizes.add(len(values))
        components[key] = dict(unit=UNITS[key], scale=max(float(values.quantile(.95)), floors[key]),
                               bound=max(float(values.quantile(.99)), floors[key]),
                               numerical_floor=floors[key])
    if len(sizes) != 1:
        raise ValueError('Unpaired calibration errors')
    return dict(version='gt_adjacent_history_v1', fps=fps, observations=sizes.pop(),
                scale_quantile=.95, bound_quantile=.99, components=components)


def choose_candidates(errors, starts, nominal, allowed, calibration):
    """Reject rather than return a usable index if no allowed candidate fits.

    Every physical component must pass; a low mean cannot hide one severe
    mismatch. Exact score ties prefer nominal proximity then the earlier index.
    Returned -1 means fallback is required; it is never a Python slice index.
    """
    shape = errors['pose'].shape
    if len(shape) != 2 or allowed.shape != shape or allowed.dtype != torch.bool:
        raise ValueError('Expected errors/allowed [B,K]')
    starts = torch.as_tensor(starts, device=allowed.device, dtype=torch.long)
    if starts.shape != (shape[1],) or (starts < 2).any() or len(starts.unique()) != len(starts):
        raise ValueError('Invalid candidate indices')
    if calibration['version'] != 'gt_adjacent_history_v1':
        raise ValueError('Unknown calibration')
    costs = torch.zeros(shape, device=allowed.device)
    compatible = allowed.clone()
    component_pass = {}
    for key, unit in UNITS.items():
        v, c = errors[key], calibration['components'][key]
        if (v.shape != shape or not torch.isfinite(v).all() or (v < 0).any()
                or c['unit'] != unit or not math.isfinite(c['scale']) or c['scale'] <= 0
                or not math.isfinite(c['bound']) or c['bound'] < 0):
            raise ValueError('Invalid physical errors or calibration')
        costs += (v/c['scale']).square()/len(UNITS)
        component_pass[key] = v <= c['bound']
        compatible &= component_pass[key]
    order = sorted(range(len(starts)), key=lambda i: (abs(int(starts[i])-nominal), int(starts[i])))
    ranked = costs[:, order].masked_fill(~compatible[:, order], float('inf'))
    winner = ranked.argmin(-1)
    indices = torch.tensor(order, device=allowed.device)[winner]
    accepted = compatible.any(-1)
    selected = torch.where(accepted, starts[indices], torch.full_like(indices, -1))
    return dict(accepted=accepted, selected_start=selected, candidate_costs=costs,
                candidate_compatible=compatible, component_pass=component_pass)


@torch.no_grad()
def propose_continuation(rep, teacher, nominal, generated_tail, target_length,
                         calibration, *, search_radius, candidate_allowed):
    """Score physical history; return targets only if the entire batch passes.

    A caller can split accepted rows explicitly or choose teacher fallback for
    the batch. An incompatible generated prefix must not be kept with that
    teacher fallback. Decoder/codec seam validation remains a separate gate.
    """
    if (teacher.ndim != 3 or generated_tail.ndim != 3 or generated_tail.shape[1] < 2
            or teacher.shape[0] != generated_tail.shape[0] or nominal < 2
            or target_length < 2 or nominal+target_length > teacher.shape[1]
            or search_radius < 0 or float(rep.fps) != calibration['fps']):
        raise ValueError('Invalid continuation input/calibration fps')
    starts = list(range(max(2, nominal-search_radius),
                        min(teacher.shape[1]-target_length, nominal+search_radius)+1))
    accepted_geometry = rep.inverse(generated_tail[:, -2:], is_normalized=False)
    errors = {key: [] for key in UNITS}
    targets = []
    for start in starts:
        history, _ = continuation_target(rep, teacher[:, start-2:start],
                                          teacher[:, start-1], generated_tail[:, -1])
        difference = history_difference(rep.inverse(history, is_normalized=False),
                                        accepted_geometry, fps=rep.fps)
        for key in UNITS:
            errors[key].append(difference[key])
        target, _ = continuation_target(rep, teacher[:, start:start+target_length],
                                         teacher[:, start-1], generated_tail[:, -1])
        targets.append(target)
    errors = {k: torch.stack(v, 1) for k, v in errors.items()}
    result = choose_candidates(errors, starts, nominal, candidate_allowed, calibration)
    result.update(starts=starts, errors=errors, target=None)
    if result['accepted'].all():
        indices = (result['selected_start'][:, None] == torch.tensor(starts, device=teacher.device)).long().argmax(1)
        result['target'] = torch.stack(targets, 1)[torch.arange(len(teacher), device=teacher.device), indices]
    return result
