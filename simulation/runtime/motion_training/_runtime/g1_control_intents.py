"""Versioned intent compiler to existing soft root goals, never edits outputs."""
import math,copy
import numpy as np
VERSION='g1_joint_interaction_v1'
def compile_intent(control,state,origin,velocity,heading,source_start):
    if not control or control.get('version')!=VERSION:
        return control,{},dict(kind='legacy',calibrated=False)
    kind=control.get('intent')
    allowed={'version','intent','velocity_xz','heading','event_id'}
    if set(control)-allowed:raise ValueError('unsupported intent field')
    if kind not in ('move','decelerate','hold','restart','release'):raise ValueError('unsupported intent')
    if kind=='release':return {},{},dict(kind=kind)
    if not np.isfinite([*origin,*velocity,heading]).all():raise ValueError('nonfinite accepted state')
    signature={k:v for k,v in control.items() if k!='heading'}
    state=copy.deepcopy(state or {})
    continuing=state.get('signature')==signature
    if kind=='hold' and not state.get('stop_anchor'):
        raise ValueError('hold requires a previously planned stopping anchor')
    if not continuing and kind!='hold':
        target=control.get('velocity_xz',[0.,0.] if kind=='decelerate' else None)
        if target is None or len(target)!=2 or not np.isfinite(target).all() or max(map(abs,target))>5:
            raise ValueError('finite target velocity required')
        if kind=='decelerate' and np.linalg.norm(target)>1e-8:raise ValueError('decelerate target must be zero')
        state=dict(signature=signature,start=source_start,origin=list(origin),v0=list(velocity),v1=list(target),heading=float(heading),transition_frames=52)
        if kind=='decelerate':
            # Discrete integral matches frame targets exactly; later drift never
            # changes this world anchor. The trajectory remains a soft condition.
            state['stop_anchor']=(np.asarray(origin)+np.asarray(velocity)*sum(1-(i+1)/52 for i in range(52))/25).tolist()
    if 'heading' in control:
        if not math.isfinite(control['heading']):raise ValueError('nonfinite heading')
        state['target_heading']=float(control['heading'])
    points=[];duration=state['transition_frames'];t0=source_start-state['start']
    for i in range(52):
        n=t0+i+1
        if kind=='hold' or (state.get('stop_anchor') is not None and n>=duration):xz=state['stop_anchor']
        else:
            ramp=min(n,duration);weight=ramp*(ramp+1)/(2*duration)
            xz=(np.asarray(state['origin'])+(np.asarray(state['v0'])*(ramp-weight)+np.asarray(state['v1'])*(weight+max(n-duration,0)))/25).tolist()
        point=dict(frame=i,xz=xz)
        if state.get('stop_anchor') is not None or 'target_heading' in state:
            h0=state['heading'];ht=state.get('target_heading',h0)
            delta=math.atan2(math.sin(ht-h0),math.cos(ht-h0))
            point['heading']=h0+delta*min(n/duration,1)
        points.append(point)
    return {'root2d':points},state,dict(kind=('hold' if state.get('stop_anchor') is not None and t0>=duration else kind),source_start=source_start,stop_anchor=state.get('stop_anchor'),calibrated=False)
