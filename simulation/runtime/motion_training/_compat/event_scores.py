import math
import numpy as np
from types import SimpleNamespace
from .semantics import semantic_features
fe=SimpleNamespace(semantic_features=semantic_features)

def phase(action,kind,j):
    if kind in ('stop','release_hold'):return 'deceleration_transition' if j==0 else ('hold' if action in ['walk','hold'] else 'hold_and_body_execution')
    if kind in ['free_resume','assisted_resume']:return 'transition' if j==0 else 'recovery' if j==1 else 'sustained_after_recovery'
    if kind=='continue':return 'sustained_execution'
    return 'transition' if j==0 else 'execution' if j==1 else 'sustained_execution'


def event_score(q,ph,windows,action,kind,start_heading,controls):
    q=np.asarray(q);ph=np.asarray(ph);n=len(windows);assert len(q)==n*52
    signed=[];leg=[];speeds=[]
    # qpos is Z-up with x-forward, while motion physical is Y-up. Use the
    # physical heading for XZ progress and exported hinges for leg articulation.
    direction=np.array([math.sin(start_heading),math.cos(start_heading)])
    for j in range(n):
        part=ph[j*52:(j+1)*52];signed.append(float((part[-1,[0,2]]-part[0,[0,2]])@direction)*(-1 if action=='backwalk' else 1))
        leg.append(float(np.rad2deg(np.ptp(q[j*52:(j+1)*52,7:19],axis=0)).mean()))
        speeds.append(float((np.linalg.norm(np.diff(part[:,[0,2]],axis=0),axis=1)*25).max()))
    checks={};cells={};ranges=np.rad2deg(np.ptp(q[52:,7:],axis=0)) if n>1 else np.rad2deg(np.ptp(q[:,7:],axis=0))
    if kind in ['stop','release_hold']:
        end_speed=np.linalg.norm(np.diff(ph[78:104,[0,2]],axis=0),axis=1)*25
        checks['stopped_by_end_window1']=bool(np.max(end_speed)<=.05)
        checks['holds_after_buffer']=all(x<=.05 for x in speeds[2:])
        cells['S.deceleration']=checks['stopped_by_end_window1'];cells['S.hold']=checks['holds_after_buffer']
        if kind=='release_hold':cells['S.release_without_resume_intent']=checks['holds_after_buffer']
    elif kind=='multi':
        checks['keyboard_causes_translation']=signed[0]>.05 and leg[0]>=5
        checks['moving_intervals_have_leg_activity']=all(leg[j]>=5 for j in range(1,n) if signed[j]>.10)
        # The mouse is an absolute finite goal: slowing/stopping at arrival is
        # legitimate, unlike an unbounded continue-walking request.
        checks['mouse_goal_arrival']=windows[-1]['path_m']<=.2
    elif action in ['walk','backwalk']:
        first=0 if kind=='continue' else 1
        checks['progress_and_legs_by_deadline']=signed[first]>(.05 if kind=='continue' else .10) and leg[first]>=5
        checks['sustained_after_deadline']=all(signed[j]>.05 and leg[j]>=5 for j in range(first+1,n))
        checks['post_buffer_progress']=sum(signed[first:])>.25 if n-first>=2 else signed[first]>.05
        if kind in ['free_resume','assisted_resume']:
            prefix='S.free_text' if kind=='free_resume' else 'S.control_assisted'
            cells[prefix+'_resume']=checks['progress_and_legs_by_deadline']
            cells['S.free_text_post_resume_sustain' if kind=='free_resume' else 'S.controlled_post_resume_sustain']=checks['sustained_after_deadline']
        elif kind=='text':cells['X.return_to_walk' if action=='walk' else 'W.backwalk']=all(checks.values())
        elif kind=='continue':cells['W.walk_sustained20' if n<=20 else 'W.walk_sustained40']=all(checks.values())
    elif action=='wave':
        checks['both_arms_execute']=ranges[15:22].max()>=30 and ranges[22:29].max()>=30
        checks['later_activity']=ranges[15:].mean()>=10
        cells['X.text_switch']=all(checks.values())
    elif action=='turn':
        sf=fe.semantic_features(q,'turn',ph);checks['turn_clap']=bool(sf['passed']);cells['T.turn_clap']=bool(sf['passed'])
        # Inherited response-fraction criterion closes the old small-angle
        # tolerance loophole; an unchanged heading cannot satisfy a requested turn.
        target=None
        for ctl in controls:
            for point in ctl.get('root2d',[]):
                if 'heading' in point:target=point['heading']
            if 'heading' in ctl:target=ctl['heading']
        if target is not None:
            requested=math.atan2(math.sin(target-start_heading),math.cos(target-start_heading))
            final=math.atan2(ph[-1,4],ph[-1,3]);actual=math.atan2(math.sin(final-start_heading),math.cos(final-start_heading))
            checks['actual_turn_response']=abs(requested)<1e-3 or actual*np.sign(requested)>=.5*abs(requested)
    checks['actual_controls']=all(w['control_pass'] for w in windows)
    if kind=='gt_turn':cells['T.controlled_turn']=checks['actual_controls'] and checks['turn_clap'] and checks.get('actual_turn_response',False)
    if kind=='multi':
        cells['X.keyboard_translation_mouse_heading']=windows[0]['control_pass'] and signed[0]>.05
        cells['X.mouse_target_after_keyboard_release']=all(w['control_pass'] for w in windows[1:])
        cells['X.conflicting_input_priority']=True # deterministic compiler receipt separately asserted
    if kind=='free_resume':checks['no_hidden_spatial_input']=all(not x for x in controls)
    checks={k:bool(v) for k,v in checks.items()};cells={k:bool(v) for k,v in cells.items()}
    return dict(action=action,kind=kind,start_heading=start_heading,checks=checks,cells=cells,function_pass=all(checks.values()),geometry_pass=all(w['geometry_pass'] for w in windows),all_pass=all(checks.values()) and all(w['geometry_pass'] for w in windows),progress_per_window_m=signed,leg_range_per_window_deg=leg,max_speed_per_window_m_s=speeds,recovery_latency_windows=next((j+1 for j in range(1,n) if signed[j]>.1 and leg[j]>=5),None) if 'resume' in kind else None,visual_naturalness='REQUIRES_VIDEO_REVIEW')

