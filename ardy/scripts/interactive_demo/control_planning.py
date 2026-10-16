"""World-space mouse deadlines and conservative live-splice checks."""
import math
import numpy as np


def waypoint_route(origin, goal, start_frame, minimum_frames, *, speed=.8, acceleration=.6):
    """Sparse smooth path, with peak speed/acceleration bounded by its duration.

    This defines soft generation conditions, never edits generated poses.
    Keep intermediate goals visible even when the final goal is beyond F196.
    """
    origin, goal = np.asarray(origin,dtype=float), np.asarray(goal,dtype=float)
    distance = float(np.linalg.norm(goal-origin))
    if not np.isfinite([*origin,*goal,distance]).all():
        raise ValueError('nonfinite mouse waypoint')
    seconds = max(float(minimum_frames)/25, 1.5*distance/speed, math.sqrt(6*distance/acceleration))
    frames = max(1,int(math.ceil(seconds*25)))
    offsets = sorted(set([0,*range(52,frames,52),frames]))
    points=[]
    for i in offsets:
        u=i/frames
        xz=origin+(goal-origin)*(3*u*u-2*u*u*u)
        points.append(dict(frame=int(start_frame+i),xz=xz.tolist()))
    return points


def live_seam_limits(previous, incoming, fps=50):
    """Dynamic continuity limits in addition to the existing coarse guard."""
    if len(previous)<2:
        raise ValueError('replacement needs two immutable samples')
    root_step=float(np.linalg.norm(previous[-1,:3]-previous[-2,:3]))
    delta=lambda a,b:np.arctan2(np.sin(a-b),np.cos(a-b))
    joint_step=np.abs(delta(previous[-1,7:],previous[-2,7:]))
    root_limit=min(.08,max(.04,root_step*1.75+.015))
    joint_limits=np.minimum(.2,np.maximum(.08,joint_step*1.75+.04))
    root_jump=float(np.linalg.norm(incoming[0,:3]-previous[-1,:3]))
    joint_jumps=np.abs(delta(incoming[0,7:],previous[-1,7:]))
    if root_jump>root_limit or np.any(joint_jumps>joint_limits):
        raise ValueError(f'unsafe replacement seam: dynamic root={root_jump:.3f}/{root_limit:.3f}m, '
                         f'joint_excess={float(np.max(joint_jumps-joint_limits)):.3f}rad')
    return dict(root_speed_mps=root_jump*fps,root_jump_limit_m=root_limit,
                maximum_joint_speed_rps=float(joint_jumps.max()*fps))
