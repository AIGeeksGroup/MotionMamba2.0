'Continuation target.'
import torch

def continuation_target(motion_rep, teacher_segment, teacher_boundary, generated_boundary):
    if teacher_segment.ndim != 3 or teacher_boundary.ndim != 2:
        raise ValueError('Expected [B,T,D] target and [B,D] boundaries')
    if generated_boundary.shape != teacher_boundary.shape:
        raise ValueError('Teacher/generated boundary shapes differ')
    if not all((torch.isfinite(t).all() for t in (teacher_segment, teacher_boundary, generated_boundary))):
        raise ValueError('Nonfinite continuation target input')
    teacher_heading = torch.atan2(teacher_boundary[:, 4], teacher_boundary[:, 3])
    generated_heading = torch.atan2(generated_boundary[:, 4], generated_boundary[:, 3])
    delta = torch.atan2(torch.sin(generated_heading - teacher_heading), torch.cos(generated_heading - teacher_heading))
    local = motion_rep.translate_2d(teacher_segment, -teacher_boundary[:, [0, 2]])
    aligned = torch.cat([motion_rep.rotate(local[i:i + 1], delta[i:i + 1]) for i in range(local.shape[0])], dim=0)
    return (aligned, delta)

@torch.no_grad()
def phase_compatible_continuation_target(motion_rep, teacher_motion, teacher_start, generated_tail, target_length, *, search_radius=8, pose_weight=1.0, velocity_weight=0.5, contact_weight=0.1):
    if teacher_motion.ndim != 3 or generated_tail.ndim != 3:
        raise ValueError('Expected teacher motion and generated tail [B,T,D]')
    if generated_tail.shape[1] < 2:
        raise ValueError('Generated tail needs at least two accepted frames')
    if teacher_motion.shape[0] != generated_tail.shape[0]:
        raise ValueError('Teacher/generated batch sizes differ')
    if target_length < 2 or search_radius < 0:
        raise ValueError('Target length must be at least two and radius nonnegative')
    if teacher_start < 2 or teacher_start + target_length > teacher_motion.shape[1]:
        raise ValueError('Nominal teacher target start is outside the source clip')
    if not all((torch.isfinite(x).all() for x in (teacher_motion, generated_tail))):
        raise ValueError('Nonfinite continuation target input')
    batch = teacher_motion.shape[0]
    device = teacher_motion.device
    starts = [start for start in range(max(2, teacher_start - search_radius), min(teacher_motion.shape[1] - target_length, teacher_start + search_radius) + 1)]
    if not starts:
        raise ValueError('No valid phase-search candidates')
    starts.sort(key=lambda start: (abs(start - teacher_start), start))
    accepted = motion_rep.inverse(motion_rep.normalize(generated_tail), is_normalized=True)
    accepted_pose = accepted['posed_joints']
    accepted_rel = accepted_pose - accepted_pose[:, :, :1]
    accepted_velocity = accepted_pose[:, 1:] - accepted_pose[:, :-1]
    accepted_rel_velocity = accepted_velocity - accepted_velocity[:, :, :1]
    accepted_contacts = accepted['foot_contacts']
    accepted_last = accepted_rel[:, -1]
    accepted_last_velocity = accepted_rel_velocity[:, -1]
    accepted_last_contacts = accepted_contacts[:, -1].bool()
    candidate_segments = []
    candidate_costs = []
    for start in starts:
        candidate = teacher_motion[:, start:start + target_length]
        teacher_boundary = teacher_motion[:, start - 1]
        (aligned, _) = continuation_target(motion_rep, candidate, teacher_boundary, generated_tail[:, -1])
        (aligned_history, _) = continuation_target(motion_rep, teacher_motion[:, start - 2:start], teacher_boundary, generated_tail[:, -1])
        geometry = motion_rep.inverse(motion_rep.normalize(aligned_history), is_normalized=True)
        pose = geometry['posed_joints']
        rel = pose - pose[:, :, :1]
        velocity = pose[:, 1:] - pose[:, :-1]
        rel_velocity = velocity - velocity[:, :, :1]
        pose_error = (rel[:, -1] - accepted_last).square().mean(dim=(-1, -2))
        velocity_error = (rel_velocity[:, 0] - accepted_last_velocity).square().mean(dim=(-1, -2))
        contacts = geometry['foot_contacts'][:, -1].bool()
        contact_error = (contacts ^ accepted_last_contacts).float().mean(dim=-1)
        cost = pose_weight * pose_error + velocity_weight * velocity_error + contact_weight * contact_error
        candidate_segments.append(aligned)
        candidate_costs.append(cost)
    costs = torch.stack(candidate_costs, dim=1)
    choice = costs.argmin(dim=1)
    segments = torch.stack(candidate_segments, dim=1)
    rows = torch.arange(batch, device=device)
    selected = segments[rows, choice]
    selected_starts = torch.tensor(starts, device=device, dtype=torch.long)[choice]
    return (selected, selected_starts, costs[rows, choice])
