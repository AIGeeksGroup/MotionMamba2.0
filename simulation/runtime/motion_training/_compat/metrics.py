import numpy as np
import torch
from ardy.exports.mujoco import MujocoQposConverter

@torch.no_grad()
def metrics(rep, explicit, target=None, boundary_offset=0):
    decoded = rep.inverse(explicit, is_normalized=True)
    p = decoded['posed_joints']
    pos = rep.inverse(explicit, is_normalized=True, posed_joints_from='positions')['posed_joints']
    feet = p[:, :, list(rep.skeleton.foot_joint_idx)]
    speed = torch.linalg.vector_norm(feet[:, 1:, :, [0, 2]]-feet[:, :-1, :, [0, 2]], dim=-1)*25
    contact = decoded['foot_contacts'][:, 1:]
    q = np.asarray(MujocoQposConverter(rep.skeleton).dict_to_qpos(decoded, device=explicit.device, numpy=True)).reshape(-1, 36)
    d = np.abs(np.arctan2(np.sin(np.diff(q[:, 7:], axis=0)), np.cos(np.diff(q[:, 7:], axis=0))))*180/np.pi
    boundary = ((np.arange(1, len(q))+boundary_offset)%52)==0
    def avg(x):
        return float(x.float().mean()) if x.numel() else None
    result = dict(frames=len(q), jitter_m_per_frame2=avg(torch.linalg.vector_norm(p[:, 2:]-2*p[:, 1:-1]+p[:, :-2], dim=-1)),
        contact_fraction=avg(contact), legacy_skate_percent=100*avg((speed>.15)&contact),
        contact_slip_percent=100*avg((speed>.15)[contact]) if contact.any() else None,
        contact_speed_m_s=avg(speed[contact]),
        root_speed_m_s=avg(torch.linalg.vector_norm(decoded['root_positions'][:, 1:, [0,2]]-decoded['root_positions'][:, :-1, [0,2]], dim=-1)*25),
        position_rotation_disagreement_cm=100*avg(torch.linalg.vector_norm(p[:,:,1:]-pos[:,:,1:], dim=-1)),
        joint_step_p95_deg=float(np.percentile(d, 95)), joint_step_max_deg=float(d.max()),
        boundary_jump_max_deg=float(d[boundary].max()) if boundary.any() else None,
        interior_jump_max_deg=float(d[~boundary].max()))
    if target is not None:
        gt = rep.inverse(target, is_normalized=True)
        gc = gt['foot_contacts'][:, 1:]
        result.update(mpjpe_cm=100*avg(torch.linalg.vector_norm(p-gt['posed_joints'], dim=-1)),
                      gt_contact_speed_m_s=avg(speed[gc]),
                      gt_contact_slip_percent=100*avg((speed>.15)[gc]) if gc.any() else None)
    return result, q

