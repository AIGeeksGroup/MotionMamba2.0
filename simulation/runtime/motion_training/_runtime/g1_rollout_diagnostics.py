"""Raw qpos diagnostics for drift and window seams; no reference correction."""
import numpy as np


def partition_foot_motion(points, qpos, *, fps=25, horizon=52):
    """Partition every adjacent-frame pair without cutting or joining trajectories.

    Uses the existing near-floor proxy, not predicted contact or physical force.
    Excess-distance sums count each virtual probe separately; they are diagnostic
    exposure measures, not the physical distance travelled by an entire foot.
    """
    p, q = np.asarray(points,dtype=np.float64), np.asarray(qpos,dtype=np.float64)
    if p.shape != (len(q),2,4,3) or q.shape != (len(q),36) or len(q)<3:
        raise ValueError('Expected matching sole[T,2,4,3] and qpos[T>=3,36]')
    if not np.isfinite(p).all() or not np.isfinite(q).all() or fps<=0 or horizon<2:
        raise ValueError('Invalid geometry or temporal convention')
    end=np.arange(1,len(q))
    masks={'first_window':end<horizon,'seams':end%horizon==0,
           'later_interiors':(end>=horizon)&(end%horizon!=0)}
    assert np.all(sum(m.astype(int) for m in masks.values())==1)
    speed=np.linalg.norm(np.diff(p,axis=0)[...,:2],axis=-1)*fps
    ground=(p[1:,...,2]<.02)&(p[:-1,...,2]<.02)
    root_velocity=np.diff(q[:,:2],axis=0)*fps
    # Acceleration at end-frame k compares (k-2 -> k-1) and (k-1 -> k).
    # A seam can affect two acceleration samples, so do not interpret only k=52
    # as the entire seam discontinuity. Save both on the seam row below.
    acceleration=np.linalg.norm(np.diff(root_velocity,axis=0),axis=-1)*fps
    def summary(mask):
        values=speed[mask][ground[mask]]
        count=int(ground[mask].sum())
        intervals=int(mask.sum())
        return dict(intervals=intervals,near_floor_probe_pairs=count,
            near_floor_fraction=count/(intervals*8) if intervals else None,
            near_floor_speed_m_s=float(values.mean()) if count else None,
            near_floor_slip_fraction=float((values>.15).mean()) if count else None,
            probe_excess_distance_m=float(np.maximum(values-.15,0).sum()/fps),
            root_speed_m_s=float(np.linalg.norm(root_velocity[mask],axis=-1).mean()) if intervals else None,
            root_acceleration_m_s2=float(acceleration[mask[1:]].mean()) if mask[1:].any() else None)
    parts={name:summary(mask) for name,mask in masks.items()}
    all_values=summary(np.ones(len(end),dtype=bool))
    for value in parts.values():
        value['share_of_total_probe_excess']=value['probe_excess_distance_m']/all_values['probe_excess_distance_m'] if all_values['probe_excess_distance_m'] else None
    windows=[]
    for start in range(0,len(q)-1,horizon):
        windows.append(dict(window=start//horizon+1,start=start,end=min(start+horizon,len(q)),
            **summary((end>start)&(end<min(start+horizon,len(q))))))
    seams=[]
    for k in end[end%horizon==0]:
        affected=[j for j in [int(k)-2,int(k)-1] if 0<=j<len(acceleration)]
        seams.append(dict(frame=int(k),root_acceleration_max_m_s2=float(acceleration[affected].max())))
    return dict(all=all_values,partitions=parts,windows=windows,seams=seams)


def summarize_qpos(qpos, *, fps=25, horizon=52):
    q = np.asarray(qpos, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 36 or len(q) < 3 or not np.isfinite(q).all():
        raise ValueError('Expected finite G1 qpos[T>=3,36]')
    if fps <= 0 or horizon < 2:
        raise ValueError('Invalid temporal convention')
    norm = np.linalg.norm(q[:, 3:7], axis=1)
    if (norm < 1e-8).any():
        raise ValueError('Invalid zero root quaternion')
    quat = q[:, 3:7] / norm[:, None]
    translation = np.linalg.norm(np.diff(q[:, :2], axis=0), axis=-1)
    joint_delta = np.rad2deg(np.diff(q[:, 7:], axis=0))
    joint_step = np.sqrt(np.mean(joint_delta**2, axis=1))
    rotation_step = np.rad2deg(2*np.arccos(np.clip(np.abs(np.sum(quat[1:]*quat[:-1], axis=1)), 0, 1)))
    # diff[k-1] connects frame k-1 to k. The first new-window frame is k=52.
    boundaries = np.arange(horizon, len(q), horizon)
    boundary_mask = np.zeros(len(q)-1, dtype=bool)
    boundary_mask[boundaries-1] = True
    def stats(values):
        return dict(mean=float(np.mean(values)), p95=float(np.percentile(values, 95)), max=float(np.max(values))) if len(values) else None
    def split(values):
        return dict(all=stats(values), seams=stats(values[boundary_mask]), interior=stats(values[~boundary_mask]))
    w, x, y, z = quat.T
    yaw = np.unwrap(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z)))
    velocity = np.diff(q[:, :2], axis=0)*fps
    direction = np.stack((np.cos(yaw[:-1]), np.sin(yaw[:-1])), axis=1)
    forward_speed = np.sum(velocity*direction, axis=1)
    left_speed = -velocity[:, 0]*direction[:, 1] + velocity[:, 1]*direction[:, 0]
    windows = []
    for start in range(0, len(q)-1, horizon):
        end = min(start+horizon, len(q))
        windows.append(dict(start_frame=start, end_frame_exclusive=end,
            root_speed_m_s=float(translation[start:end-1].mean()*fps),
            root_height_min_m=float(q[start:end, 2].min()),
            root_height_max_m=float(q[start:end, 2].max())))
    return dict(frames=len(q), fps=fps, horizon=horizon,
        root_displacement_from_start_max_m=float(np.linalg.norm(q[:, :2]-q[:1, :2], axis=1).max()),
        root_displacement_end_m=float(np.linalg.norm(q[-1, :2]-q[0, :2])),
        root_path_m=float(translation.sum()),
        root_yaw_total_deg=float(np.rad2deg(yaw[-1]-yaw[0])),
        signed_forward_speed_m_s=float(forward_speed.mean()),
        signed_left_speed_m_s=float(left_speed.mean()),
        root_speed=split(translation*fps),
        joint_step_rms_deg=split(joint_step),
        root_rotation_step_deg=split(rotation_step),
        quaternion_norm_max_error=float(np.abs(norm-1).max()),
        seams=[dict(frame=int(k), root_step_m=float(translation[k-1]),
                    joint_step_rms_deg=float(joint_step[k-1]),
                    root_rotation_step_deg=float(rotation_step[k-1])) for k in boundaries],
        windows=windows,
        note='Diagnostic only. Heading-projected velocities use G1 qpos root X as forward; semantics require source/video review.')
