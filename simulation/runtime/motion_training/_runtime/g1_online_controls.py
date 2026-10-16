"""Bounded sparse controls in ARDY world coordinates, before denoising.

One window = 52 source frames at 25 Hz. Positions are metres in world XZ
(Y up); forward=(sin(h),cos(h)), h=atan2(vx,vz), radians.
MuJoCo conversion happens later. Heading-only means orientation without a
position target. Global feature channels remain [cos(h),sin(h)].
Spatial goals and text are both soft CFG conditions; neither guarantees hard
precedence when they conflict. The viewer explicitly warns about this case.
"""
import hashlib
import json
import math

import torch
from ardy.constraints import Root2DConstraintSet


def control_fingerprint(control):
    return hashlib.sha256(json.dumps(control or {}, sort_keys=True, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


def conditions(model, control, origin):
    control = control or {}
    if set(control) - {'root2d', 'velocity_xz', 'heading', 'version'}:
        raise ValueError('unsupported control field')
    device = model.device
    points = control.get('root2d', [])
    if not isinstance(points, list) or len(points) > 52:
        raise ValueError('root2d must contain at most 52 sparse points')
    velocity = control.get('velocity_xz')
    heading_only = velocity is None and not points and 'heading' in control
    if 'heading' in control and (not isinstance(control['heading'], (float, int)) or not math.isfinite(control['heading'])):
        raise ValueError('heading must be finite')
    if heading_only:
        anchor = origin[0, [0, 2]].detach().cpu().tolist()
        points = [{'frame': i, 'xz': anchor, 'heading': control['heading']} for i in range(52)]
    if velocity is not None and points:
        raise ValueError('choose velocity or waypoints, not both')
    if velocity is not None:
        if len(velocity) != 2 or any(not math.isfinite(v) or abs(v) > 5 for v in velocity):
            raise ValueError('velocity_xz must be two finite m/s values within +/-5')
        anchor = origin[0, [0, 2]].detach().cpu().tolist()
        points = [{'frame': i, 'xz': [anchor[j] + velocity[j] * (i + 1) / 25 for j in range(2)]}
                  for i in range(52)]
    frames, positions, headings = [], [], []
    for point in points:
        if set(point) - {'frame', 'xz', 'heading'}:
            raise ValueError('unsupported waypoint field')
        frame = point['frame']
        if type(frame) is not int or not 0 <= frame < 52 or frame in frames:
            raise ValueError('waypoint frames must be unique integers in [0,52)')
        xz = point['xz']
        if len(xz) != 2 or any(not math.isfinite(v) or abs(v) > 100000 for v in xz):
            raise ValueError('waypoint xz must contain finite world coordinates')
        frames.append(frame)
        positions.append(xz)
        headings.append(point.get('heading', control.get('heading')))
    if any(h is not None for h in headings) and not all(h is not None and math.isfinite(h) for h in headings):
        raise ValueError('heading must be finite and specified for all controlled points')
    if not frames:
        return torch.zeros(1, 52, 414, device=device), torch.zeros(1, 52, 414, dtype=torch.bool, device=device)
    constraint = Root2DConstraintSet(model.skeleton, torch.tensor(frames, device=device),
        torch.tensor(positions, device=device, dtype=torch.float32),
        torch.tensor(headings, device=device, dtype=torch.float32) if headings[0] is not None else None)
    observed, mask = model.motion_rep.create_conditions_from_constraints(
        [constraint], length=52, to_normalize=False, device=device)
    if heading_only:
        # Use the authoritative constraint encoder, then remove position goals.
        mask[:, [0, 2]] = False
    observed = model.motion_rep.normalize(observed) * mask
    return observed[None], mask[None].bool()
