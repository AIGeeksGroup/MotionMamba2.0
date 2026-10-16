import numpy as np
from .torso import torso_pitch

def semantic_features(q,action,physical,reference=None):
    q=np.asarray(q);ranges=np.rad2deg(np.ptp(q[:,7:],axis=0));speed=float(np.linalg.norm(np.diff(q[:,:2],axis=0),axis=1).mean()*25)
    h=np.unwrap(np.arctan2(physical[:,4],physical[:,3]));angle=float(np.rad2deg(h[-1]-h[0]));w,x,y,z=q[0,3:7];direction=np.array([1-2*(y*y+z*z),2*(x*y+w*z)])
    signed=float((q[-1,:2]-q[0,:2])@direction);arms=float(ranges[15:].mean());legs=float(ranges[:12].mean());height=float(np.ptp(q[:,2]))
    checks={}
    if action in ('walk','backwalk'):
        checks=dict(progress=signed>.10 if action=='walk' else signed<-.10,leg_articulation=legs>=5)
    elif action=='turn':checks=dict(turn_excursion=abs(angle)>=8,arm_articulation=arms>=5,leg_articulation=legs>=3)
    elif action=='wave':checks=dict(left_arm_motion=float(ranges[15:22].max())>=15,right_arm_motion=float(ranges[22:29].max())>=15,upper_body_activity=arms>=5)
    elif action=='squat':checks=dict(bilateral_knee_motion=min(float(ranges[3]),float(ranges[9]))>=10,height_change=height>=.025)
    elif action=='leg_raise':checks=dict(leg_excursion=float(max(ranges[0],ranges[6]))>=20,knee_excursion=float(max(ranges[3],ranges[9]))>=15)
    elif action=='rhythm':checks=dict(bilateral_arm_motion=min(float(ranges[15:22].max()),float(ranges[22:29].max()))>=15,upper_body_activity=arms>=5)
    elif action=='bend':
        pitch=torso_pitch(q);checks=dict(torso_excursion=float(np.ptp(pitch))>=10,hip_excursion=max(float(ranges[0]),float(ranges[6]))>=8)
    return dict(checks=checks,passed=bool(checks and all(checks.values())),speed_m_s=speed,signed_displacement_m=signed,
      heading_change_deg=angle,arm_range_deg=arms,leg_range_deg=legs,height_range_m=height,
      scope='Multi-feature gross activity screen within52 frames; no claim full repeated-cycle completion or naturalness. Full recovery checked by separate156-frame event gates.')

