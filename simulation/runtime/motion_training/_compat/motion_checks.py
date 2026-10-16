import numpy as np
from types import SimpleNamespace
from .metrics import metrics
def measure(m,dec):
    row,q=metrics(m.motion_rep,dec)
    row["leg_range_deg"]=float(np.rad2deg(np.ptp(q[:,7:19],axis=0)).mean())
    return row,q
ev=SimpleNamespace(metrics=metrics,audit=SimpleNamespace(measure=measure))

def checks(action,q,feet,contact,windows,previous=None):
 speed=float(np.linalg.norm(np.diff(q[:,:2],axis=0),axis=1).mean()*25)
 deg=np.rad2deg(np.ptp(q[:,7:],axis=0));w,x,y,z=q[0,3:7]
 forward=np.array([1-2*(y*y+z*z),2*(x*y+w*z)])
 signed=float((q[-1,:2]-q[0,:2])@forward)
 active=sum(r['leg_range_deg']>=10 for r in windows)/len(windows)
 height=float(np.ptp(q[:,2]));legs=float(deg[:12].mean());arms=float(deg[15:].mean())
 # MuJoCo qpos joint order: left leg6, right leg6, waist3, left arm7,right arm7.
 if action=='walk':semantic=speed>=.15 and active>=.75 and signed>.25
 elif action=='backwalk':semantic=signed<-.25 and active>=.75
 elif action=='wave':semantic=arms>=15 and max(deg[15:].max(),0)>=30
 elif action=='squat':semantic=min(deg[3],deg[9])>=25 and height>=.08
 elif action=='bow':semantic=max(deg[0],deg[6],deg[14])>=20
 elif action=='dance':semantic=legs>=15 and arms>=15 and active>=.5
 elif action=='kick':semantic=deg[6]>=35 and deg[6]>=1.15*deg[0]
 elif action=='jump_back':
  floor=np.quantile(feet[:,:,1],.05)
  air=(feet[:,:,1].min(1)>floor+.04)
  semantic=signed<-.05 and height>=.08 and np.count_nonzero(air)>=3
 else:raise ValueError(action)
 return dict(semantic_proxy_pass=bool(semantic),root_speed_m_s=speed,signed_forward_displacement_m=signed,
  leg_range_deg=legs,arm_range_deg=arms,root_height_range_m=height,active_window_fraction=active,
  joint_ranges_deg=deg.tolist(),semantic_scope='gross kinematic screen; not a complete action classifier')


def summarize(m,dec,rows,action,limits,previous=None):
 whole,q=ev.metrics(m.motion_rep,dec);geo=m.motion_rep.inverse(dec,is_normalized=True)
 feet=geo['posed_joints'][0,:,list(m.skeleton.foot_joint_idx)].cpu().numpy()
 contact=geo['foot_contacts'][0].cpu().numpy()
 out=checks(action,q,feet,contact,rows,previous)
 violation=np.maximum(np.maximum(limits[:,0]-q[:,7:],q[:,7:]-limits[:,1]),0)
 out.update(whole=whole,windows=rows,joint_limit_max_deg=float(np.rad2deg(violation).max()))
 failures=[]
 if not np.isfinite(q).all():failures.append('nonfinite')
 if out['joint_limit_max_deg']>2:failures.append('joint_limits')
 if whole['contact_fraction'] is None or whole['contact_fraction']<.05:failures.append('contact_coverage')
 if whole['contact_slip_percent'] is None or whole['contact_slip_percent']>10:failures.append('contact_slip')
 if previous is not None:
  _,old=ev.audit.measure(m,previous[:,-2:])
  d=np.rad2deg(np.arctan2(np.sin(q[0,7:]-old[-1,7:]),np.cos(q[0,7:]-old[-1,7:])))
  out['seam_joint_rms_deg']=float(np.sqrt((d*d).mean()));out['seam_root_m']=float(np.linalg.norm(q[0,:3]-old[-1,:3]))
  if out['seam_joint_rms_deg']>10:failures.append('joint_seam')
  if out['seam_root_m']>.1:failures.append('root_seam')
 out['geometry_failures']=failures
 return out,q,feet,contact

