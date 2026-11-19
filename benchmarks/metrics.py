"""Geometry-only Root CE, End CE and coarse command matching."""
import numpy as np

def command_matching(root,joints,command,previous_root,fps=25,threshold=0.1):
    root=np.asarray(root,dtype=np.float64);joints=np.asarray(joints,dtype=np.float64)
    previous=np.asarray(previous_root,dtype=np.float64)[None]
    velocity=np.diff(np.concatenate((previous,root),axis=0),axis=0)[:,[0,2]]*fps
    hip=joints[:,8]-joints[:,1]
    forward=np.stack((hip[:,2],-hip[:,0]),axis=1)
    forward/=np.maximum(np.linalg.norm(forward,axis=1,keepdims=True),1e-8)
    signed=(velocity*forward).sum(1);speed=np.linalg.norm(velocity,axis=1)
    if command=='forward':matched=signed>threshold
    elif command=='backward':matched=signed < -threshold
    elif command=='stand':matched=speed < threshold
    elif command in ['walk','other']:matched=speed>threshold
    else:raise ValueError(f'Unknown coarse root command: {command}')
    return float(matched.mean())

def measure(root,joints,case,commands,initial_root,window_frames=52,fps=25,threshold=0.1):
    root=np.asarray(root,dtype=np.float64);joints=np.asarray(joints,dtype=np.float64)
    if root.shape!=(len(commands)*window_frames,3) or joints.shape!=(len(root),34,3):
        raise ValueError('Unexpected rollout geometry shape')
    if not np.isfinite(root).all() or not np.isfinite(joints).all():raise ValueError('Nonfinite rollout geometry')
    root_ce=float(np.linalg.norm(root[-1,[0,2]]-case['root_goal_xz_m'])*100)
    end_ce=float(np.linalg.norm(joints[-1,case['selected_joint_index']]-case['end_goal_xyz_m'])*100)
    fractions=[]
    for i,command in enumerate(commands):
        lo=i*window_frames;hi=lo+window_frames
        prior=root[lo-1] if i else initial_root
        fractions.append(command_matching(root[lo:hi],joints[lo:hi],command,prior,fps,threshold))
    return dict(root_ce_cm=root_ce,end_ce_cm=end_ce,cmd_percent=float(np.mean(fractions)*100))

def family_interval(rows,key,repeats=1000,seed=20261021):
    families={}
    for row in rows:families.setdefault(row['source_family'],[]).append(row[key])
    means=np.asarray([np.mean(v) for v in families.values()],dtype=np.float64)
    if not len(means):raise ValueError('No trials to summarize')
    rng=np.random.default_rng(seed)
    samples=means[rng.integers(0,len(means),(repeats,len(means)))].mean(1)
    return dict(mean=float(means.mean()),ci95=np.quantile(samples,[.025,.975]).tolist(),families=len(means))
