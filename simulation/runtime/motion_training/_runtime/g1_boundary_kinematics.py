"""Physical continuity at the actual accepted-history / new-window seam.

This is a soft training objective, not an output filter or a gait-phase target
solver. History is an immutable reference; only the new prediction gets grads.
"""
import torch
import torch.nn.functional as F


def accepted_boundary_kinematics(current, history, history_contacts, target_contacts,
                                foot_indices, valid, *, fps=25):
    """Compare first generated pose against the last TWO accepted poses.

Positions are world-space [B,J,3] / [B,2,J,3], in meters, Y-up.
Root/body terms favor continuous velocity at the single boundary. Foot term
uses horizontal speed only where both sides label the same foot in contact.
Returned losses use smooth-L1(beta=0.1 m/s), not squared normalized latents.
"""
    if current.ndim != 3 or current.shape[-1] != 3:
        raise ValueError('Expected current [B,J,3]')
    if history.shape != (current.shape[0],2,*current.shape[1:]):
        raise ValueError('Expected two accepted poses in the same world frame')
    if valid.shape != current.shape[:1] or valid.dtype != torch.bool or fps <= 0:
        raise ValueError('Invalid valid-history mask or frame rate')
    if history_contacts.shape != target_contacts.shape or history_contacts.shape != (len(current),len(foot_indices)):
        raise ValueError('Contact labels must match foot indices')
    history = history.detach().to(current)
    zero = current.sum()*0
    if not valid.any():
        return dict(root=zero, body=zero, foot=zero, total=zero,
                    examples=zero.detach(), contact_pairs=zero.detach())
    prior_velocity = (history[:,1]-history[:,0])*fps
    next_velocity = (current-history[:,1])*fps
    root = F.smooth_l1_loss(next_velocity[valid,0], prior_velocity[valid,0], beta=.1)
    # Separate root translation from articulation; body loss cannot be reduced
    # by translating the entire skeleton to hide a limb discontinuity.
    body = F.smooth_l1_loss((next_velocity-next_velocity[:,:1])[valid,1:],
                           (prior_velocity-prior_velocity[:,:1])[valid,1:], beta=.1)
    contacts = history_contacts.detach().bool() & target_contacts.detach().bool() & valid[:,None]
    foot_velocity = next_velocity[:,foot_indices][:,:,[0,2]]
    foot = F.smooth_l1_loss(foot_velocity[contacts], torch.zeros_like(foot_velocity[contacts]), beta=.1) if contacts.any() else zero
    return dict(root=root, body=body, foot=foot, total=root+body+foot,
                examples=valid.sum().to(current.dtype), contact_pairs=contacts.sum().to(current.dtype))
