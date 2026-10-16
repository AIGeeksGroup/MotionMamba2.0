import math
import numpy as np
import torch
from types import SimpleNamespace
from ..io import read
from .motion_checks import measure,summarize
rt=SimpleNamespace(ev=SimpleNamespace(audit=SimpleNamespace(measure=measure)),summarize=summarize)

def score_window(m,dec,q,previous,cfg,action,ctl):
    limits=np.asarray(read(cfg['joint_limit_spec'])['bounds'])
    measured,_=rt.ev.audit.measure(m,dec)
    met,_,_,_=rt.summarize(m,dec,[measured],'dance' if action in ['turn','hold'] else action,limits,m.motion_rep.normalize(previous[:,-52:]))
    ph=m.motion_rep.unnormalize(dec)[0].detach().cpu().numpy()
    path=max((float(np.linalg.norm(ph[x['frame'],[0,2]]-x['xz'])) for x in ctl.get('root2d',[]) if 'xz' in x),default=0.)
    headings=[(x['frame'],x['heading']) for x in ctl.get('root2d',[]) if 'heading' in x]
    if 'heading' in ctl:headings.append((51,ctl['heading']))
    hd=max((abs(math.degrees(math.atan2(math.sin(math.atan2(ph[f,4],ph[f,3])-h),math.cos(math.atan2(ph[f,4],ph[f,3])-h)))) for f,h in headings),default=0.)
    slip=met['whole']['contact_slip_percent'];coverage=met['whole']['contact_fraction']
    geo=m.motion_rep.inverse(dec,is_normalized=True)
    with torch.no_grad():
        prior=m.motion_rep.inverse(m.motion_rep.normalize(previous[:,-2:]),is_normalized=True)
        feet=geo['posed_joints'][:,:,list(m.skeleton.foot_joint_idx)]
        pf=prior['posed_joints'][:,:,list(m.skeleton.foot_joint_idx)]
        floor=torch.quantile(pf[...,1].flatten(),.1)
        height=feet[...,1]-floor;speed=torch.diff(torch.cat([pf[:,-1:],feet],1),dim=1)[...,[0,2]].norm(dim=-1)*25
        near=height<.04
        diagnostics=dict(predicted_contact_fraction=coverage,near_floor_fraction=float(near.float().mean()),low_feet_sliding_fraction=float(((speed>.15)&near).float().mean()),all_feet_above15cm_fraction=float((height>.15).all(-1).float().mean()),max_penetration_m=float(torch.relu(-height).max()),geometric_low_foot_speed_m_s=float(speed[near].mean()) if near.any() else None,ground_truth_contact=False,low_feet_mask_for_diagnostics_only=True)
    valid=all(x is not None and np.isfinite(x) for x in [slip,coverage,met['joint_limit_max_deg'],met['seam_root_m'],met['seam_joint_rms_deg']])
    contact=valid and slip<=10 and coverage>=.05 and diagnostics['all_feet_above15cm_fraction']<=.05
    geometry=valid and contact and met['joint_limit_max_deg']<=2 and met['seam_root_m']<=.1 and met['seam_joint_rms_deg']<=10
    return dict(metrics=met,path_m=path,heading_deg=hd,control_pass=path<=.2 and hd<=20,joint_pass=valid and met['joint_limit_max_deg']<=2,contact_pass=bool(contact),geometry_pass=bool(geometry),valid=valid,contact_diagnostics=diagnostics),ph

