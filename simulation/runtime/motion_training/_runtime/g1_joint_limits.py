"""Differentiable G1 joint-limit objective in the actual qpos export convention."""
from functools import lru_cache
import json
from pathlib import Path
import torch


@lru_cache(maxsize=8)
def load_spec(path):
    spec=json.loads(Path(path).read_text())
    bounds=torch.tensor(spec['bounds'],dtype=torch.float32)
    limited=torch.tensor(spec['limited'],dtype=torch.bool)
    if spec.get('units')!='radian' or bounds.shape!=(29,2) or limited.shape!=(29,):
        raise ValueError('Expected 29 G1 qpos joint bounds in radians')
    if len(spec['joint_names'])!=29 or len(set(spec['joint_names']))!=29:
        raise ValueError('Expected unique joint names')
    if not torch.isfinite(bounds).all() or not (bounds[limited,0]<bounds[limited,1]).all():
        raise ValueError('Invalid joint bounds')
    return spec,bounds,limited


@lru_cache(maxsize=4)
def converter_for(skeleton):
    from ardy.exports.mujoco import MujocoQposConverter
    return MujocoQposConverter(skeleton)


def exported_joint_angles(local_rotations,converter):
    r=converter._rot_offsets_f2q.to(local_rotations) @ local_rotations
    r=r[:,:,converter._mujoco_indices_to_ardy_indices]
    axis=converter._mujoco_joint_axis_values_ardy_space.to(r)
    # Select the actual hinge axis BEFORE atan2: unused Euler components may
    # be singular, and 0 * NaN gradients would contaminate otherwise valid DOFs.
    selected=axis.abs().argmax(-1)[None,None,:,None].expand(*r.shape[:3],1)
    sine=torch.stack([r[...,2,1],r[...,0,2],r[...,1,0]],dim=-1).gather(-1,selected).squeeze(-1)
    cosine=torch.stack([r[...,2,2],r[...,0,0],r[...,1,1]],dim=-1).gather(-1,selected).squeeze(-1)
    return torch.atan2(sine,cosine)*axis.sum(-1)


def limit_penalty(angles,bounds,limited,valid):
    if not valid.any():raise ValueError('No valid generated frames')
    lo,hi=bounds.to(angles).unbind(-1)
    excess=torch.relu(lo-angles)+torch.relu(angles-hi)
    return excess[valid][:,limited.to(angles.device)].square().mean()


def joint_limit_loss(geometry,skeleton,spec_path,valid):
    spec,bounds,limited=load_spec(str(spec_path))
    converter=converter_for(skeleton)
    names=[name.replace('_skel','_joint') for name in converter._mujoco_joint_including_root_list[1:]]
    if names!=spec['joint_names']:raise ValueError('Joint-limit specification order differs from qpos exporter')
    return limit_penalty(exported_joint_angles(geometry['local_rot_mats'],converter),bounds,limited,valid)
