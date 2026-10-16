"""Differentiable G1 sole points after the actual qpos hinge projection.

Opt-in: does not modify training defaults or generation. The robot topology
must come from the independently validated MuJoCo contract exporter.
"""
import json
from pathlib import Path
import torch
from g1_joint_limits import exported_joint_angles, converter_for


class ExportedG1Sole(torch.nn.Module):
    def __init__(self, contract_path):
        super().__init__()
        c = json.loads(Path(contract_path).read_text())
        if c['version'] != 'g1_mujoco_sole_fk_v1' or len(c['joint_names']) != 29:
            raise ValueError('Unrecognized G1 sole contract')
        self.contract = c
        self.register_buffer('positions', torch.tensor([n['position'] for n in c['nodes']], dtype=torch.float64))
        self.register_buffer('rotations', torch.tensor([n['rotation'] for n in c['nodes']], dtype=torch.float64))
        self.register_buffer('probes', torch.tensor(c['probes'], dtype=torch.float64))
        self.register_buffer('axes', torch.tensor([n['joint']['axis'] if n['joint'] else [0., 0., 1.]
                                                  for n in c['nodes']], dtype=torch.float64))
        self.register_buffer('anchors', torch.tensor([n['joint']['anchor'] if n['joint'] else [0., 0., 0.]
                                                     for n in c['nodes']], dtype=torch.float64))

    def forward(self, root_position, root_rotation, angles):
        if root_position.shape[-1] != 3 or root_rotation.shape != (*root_position.shape[:-1], 3, 3):
            raise ValueError('Expected root xyz and matching rotation matrices')
        if angles.shape != (*root_position.shape[:-1], 29):
            raise ValueError('Expected 29 exported hinge angles')
        positions = {self.contract['root_body']: root_position}
        rotations = {self.contract['root_body']: root_rotation}
        eye = torch.eye(3, dtype=angles.dtype, device=angles.device)
        for index, node in enumerate(self.contract['nodes']):
            parent = node['parent']
            base_rotation = self.rotations[index].to(angles)
            offset = self.positions[index].to(angles)
            if node['joint'] is not None:
                j = node['joint']
                angle = angles[..., j['column']]-j['reference']
                axis = self.axes[index].to(angles)
                x, y, z = axis.unbind()
                zero = x*0
                skew = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero]).reshape(3, 3)
                rotation = eye + angle.sin()[..., None, None]*skew + (1-angle.cos())[..., None, None]*(skew@skew)
                anchor = self.anchors[index].to(angles)
                offset = offset + (base_rotation@(anchor-(rotation@anchor[..., None]).squeeze(-1))[..., None]).squeeze(-1)
                relative = base_rotation@rotation
            else:
                relative = base_rotation
            rotations[node['body_id']] = rotations[parent]@relative
            positions[node['body_id']] = positions[parent]+(rotations[parent]@offset[..., None]).squeeze(-1)
        output = []
        probes = self.probes.to(angles)
        for body in self.contract['foot_body_ids']:
            output.append(positions[body][..., None, :]+probes@rotations[body].transpose(-1, -2))
        return torch.stack(output, dim=-3)

    def from_geometry(self, geometry, skeleton):
        converter = converter_for(skeleton)
        names = [n.replace('_skel', '_joint') for n in converter._mujoco_joint_including_root_list[1:]]
        if names != self.contract['joint_names']:
            raise ValueError('Exporter/robot joint order mismatch')
        local = geometry['local_rot_mats']
        transform = converter.ardy_to_mujoco_matrix.to(local)
        root_rotation = transform@(converter._rot_offsets_f2q[0].to(local)@local[..., 0, :, :])@transform.T
        root_position = geometry['root_positions']@transform.T
        angles = exported_joint_angles(local, converter)
        return self(root_position, root_rotation, angles)


def contact_regions(contacts):
    """Map ankle/toe proxy labels to the two heel/two toe probes per foot."""
    if contacts.shape[-1] != 4:
        raise ValueError('Expected L_ankle,L_toe,R_ankle,R_toe contacts')
    return contacts.detach().bool().reshape(*contacts.shape[:-1], 2, 2).repeat_interleave(2, dim=-1)


def fixed_support_velocity_loss(points, reference_points, reference_contacts, *, fps=25, height=.02):
    """Actual exported sole velocity on fixed reference support probes.

Reference membership cannot change with the current prediction. No-contact
coverage is reported explicitly; this objective alone cannot certify motion.
"""
    if points.ndim != 5 or points.shape[-3:] != (2, 4, 3) or points.shape[1] < 2:
        raise ValueError('Expected [B,T,2,4,3], T>=2')
    if reference_points.shape != points.shape or reference_contacts.shape != (*points.shape[:2], 4):
        raise ValueError('Reference shape mismatch')
    reference = reference_points.detach().to(points)
    regions = contact_regions(reference_contacts)
    low = reference[..., 2] < height
    mask = regions[:, 1:] & regions[:, :-1] & low[:, 1:] & low[:, :-1]
    velocity = (points[:, 1:, ..., :2]-points[:, :-1, ..., :2])*fps
    loss = velocity[mask].square().mean() if mask.any() else points.sum()*0
    return dict(loss=loss, support_pairs=mask.sum().detach(), mask=mask,
                max_speed=velocity[mask].norm(dim=-1).max().detach() if mask.any() else None)
