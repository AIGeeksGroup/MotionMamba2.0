"""Persistent absolute-world soft goal, planned from the replaceable boundary.

This compiles conditioning only. It never changes generated qpos or claims
that a kinematically feasible reference has been followed by the model.
"""
import math
import copy
import numpy as np
VERSION='g1_absolute_goal_v2'
FPS=25

def wrap(a):return math.atan2(math.sin(a),math.cos(a))

def make_plan(origin,velocity,heading,target,target_heading=None):
    p=np.asarray(origin,dtype=float);v=np.asarray(velocity,dtype=float);goal=np.asarray(target,dtype=float)
    if np.linalg.norm(goal-p)>20 or np.linalg.norm(v)>5:
        raise ValueError('absolute goal exceeds bounded planner range (20m or incoming 5m/s)')
    points=[];angles=[];phases=[]
    # First brake without changing the user's world target. Discrete integration
    # starts with the actual accepted incoming velocity, ends at zero.
    n=max(2,int(math.ceil(1.5*np.linalg.norm(v)/.8*FPS)))
    for i in range(1,n+1):
        u=i/n;p=p+v*(1-3*u*u+2*u*u*u)/FPS
        points.append(p.copy());angles.append(heading);phases.append('brake')
    delta=goal-p;distance=float(np.linalg.norm(delta))
    facing=math.atan2(delta[0],delta[1]) if distance>1e-8 else heading
    turn=wrap(facing-heading);n=max(1,int(math.ceil(1.5*abs(turn)/math.radians(45)*FPS)))
    for i in range(1,n+1):
        u=i/n;points.append(p.copy());angles.append(heading+turn*(3*u*u-2*u*u*u));phases.append('turn')
    start=p.copy()
    # Minimum-jerk segment ends at the exact world target with zero velocity.
    duration=max(2.,1.875*distance/.8,math.sqrt(5.78*distance/.8))
    n=int(math.ceil(duration*FPS))
    for i in range(1,n+1):
        u=i/n;s=10*u**3-15*u**4+6*u**5
        points.append(start+delta*s);angles.append(facing);phases.append('return' if abs(turn)>math.pi/2 else 'approach')
    if target_heading is not None:
        turn=wrap(target_heading-facing);n=max(1,int(math.ceil(1.5*abs(turn)/math.radians(45)*FPS)))
        for i in range(1,n+1):
            u=i/n;points.append(goal.copy());angles.append(facing+turn*(3*u*u-2*u*u*u));phases.append('final_heading')
    terminal=angles[-1]
    for _ in range(52):points.append(goal.copy());angles.append(terminal);phases.append('hold')
    points=np.asarray(points);angles=np.asarray(angles)
    vs=np.diff(np.vstack((origin,points)),axis=0)*FPS
    acc=np.diff(np.vstack((velocity,vs)),axis=0)*FPS
    return dict(points=points.tolist(),headings=angles.tolist(),phases=phases,
        target_xz=goal.tolist(),terminal_velocity=[0.,0.],frames=len(points),
        max_speed=float(np.linalg.norm(vs,axis=1).max()),max_acceleration=float(np.linalg.norm(acc,axis=1).max()),
        max_heading_rate=float(np.abs(np.arctan2(np.sin(np.diff(np.r_[heading,angles])),np.cos(np.diff(np.r_[heading,angles])))).max()*FPS),
        feasible=True,scope='soft reference kinematics only; model/physics acceptance separate')

def compile_goal(control,state,origin,velocity,heading,source_start):
    assert control.get('version')==VERSION
    if control.get('intent')=='cancel':return {},{},dict(kind='cancel',source_start=source_start)
    if control.get('intent')!='absolute_target':raise ValueError('unsupported absolute goal intent')
    allowed={'version','intent','target_xz','heading','event_id'}
    if set(control)-allowed:raise ValueError('unsupported absolute goal field')
    if len(control.get('target_xz',[]))!=2 or not np.isfinite([*origin,*velocity,heading,*control['target_xz'],control.get('heading',heading)]).all():
        raise ValueError('finite accepted state and 2D absolute goal required')
    state=copy.deepcopy(state or {});signature=copy.deepcopy(control);reason=None
    if state.get('goal_signature')!=signature:reason='new_absolute_target'
    elif source_start<state['start']:raise ValueError('goal planner must restore matching snapshot before rewind')
    else:
        j=min(source_start-state['start']-1,len(state['plan']['points'])-1)
        if j>=0 and np.linalg.norm(np.asarray(origin)-state['plan']['points'][j])>.2:reason='accepted_state_off_path'
    if reason:
        state=dict(goal_signature=signature,start=int(source_start),plan=make_plan(origin,velocity,heading,control['target_xz'],control.get('heading')))
    plan=state['plan'];offset=source_start-state['start'];last=len(plan['points'])-1
    points=[dict(frame=i,xz=plan['points'][min(offset+i,last)],heading=plan['headings'][min(offset+i,last)]) for i in range(52)]
    receipt=dict(kind='absolute_target',source_start=int(source_start),absolute_target_xz=control['target_xz'],
        phase=plan['phases'][min(offset,last)],plan_start=state['start'],plan_end=state['start']+plan['frames'],
        replan_reason=reason,goal_is_behind=bool(np.dot(np.asarray(control['target_xz'])-origin,[math.sin(heading),math.cos(heading)])<0),
        target_shift_m=0.,persistent_until_cancel=True,terminal_velocity=plan['terminal_velocity'],
        max_reference_acceleration=plan['max_acceleration'],first_correct_action=None)
    return dict(root2d=points),state,receipt

def boundary_receipt(published_window,generated_window,consumed_frame):
    if published_window<0 or generated_window<published_window:raise ValueError('invalid publication frontier')
    boundary=int(published_window)*52
    if consumed_frame>=boundary and boundary:raise ValueError('consumer ahead of published frontier')
    return dict(consumed_source_frame=float(consumed_frame),published_source_boundary=boundary,
        earliest_effective_source_frame=boundary,generated_source_boundary=int(generated_window)*52,
        immutable_ahead_frames=max(0.,boundary-float(consumed_frame)),first_correct_action_source_frame=None)
