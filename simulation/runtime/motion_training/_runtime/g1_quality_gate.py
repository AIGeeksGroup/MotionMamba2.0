"""Fail-closed quality retention checks, separate from absolute acceptance."""
import math
from collections import defaultdict


def groups(rows):
    grouped=defaultdict(list)
    for row in rows:
        grouped[(row['category'],row['condition'],row['frames'])].append(row)
    return grouped


def compare_quality(rows, baseline):
    current,reference=groups(rows),groups(baseline)
    failures=[]
    comparisons=[]
    if not rows or not baseline:
        failures.append('empty panel')
    if set(current)!=set(reference):
        failures.append('panel groups differ')
    error_keys=['contact_speed_m_s','jitter_m_per_frame2','position_rotation_disagreement_cm']
    for key,old in reference.items():
        new=current.get(key,[])
        if len(new)!=len(old) or len({(r['sample'],r['seed']) for r in new})!=len(new) or {(r['sample'],r['seed']) for r in new}!={(r['sample'],r['seed']) for r in old}:
            failures.append(f'{key}: paired samples differ')
            continue
        checks=error_keys+['contact_fraction','root_speed_m_s','root_path_m']
        if key[1]=='root_goal_52':
            checks=checks+['root_goal_error_m']
        for metric in checks:
            values=[r.get(metric) for r in new]
            refs=[r.get(metric) for r in old]
            if any(v is None or not math.isfinite(v) for v in values+refs):
                failures.append(f'{key}: missing/nonfinite {metric}')
                continue
            a,b=sum(values)/len(values),sum(refs)/len(refs)
            comparisons.append(dict(group=list(key),metric=metric,current=a,baseline=b))
            # Per-group checks prevent improving only the numerous idle cases.
            floor={'contact_speed_m_s':.03,'jitter_m_per_frame2':.003,
                   'position_rotation_disagreement_cm':.3, 'root_goal_error_m':.03}.get(metric,0.)
            if metric in error_keys+['root_goal_error_m'] and a>max(b*1.25,floor):
                failures.append(f'{key}: {metric} regression {a:.5g}>{b:.5g}')
            if metric=='contact_fraction' and a<.8*b:
                failures.append(f'{key}: contact coverage collapsed')
            if metric in ['root_speed_m_s','root_path_m'] and key[0] in ['walk','slow_walk','side_walk','turn'] and a<.75*b:
                failures.append(f'{key}: movement collapsed ({metric})')
    # This is a retention/safety gate, never a claim of correct text semantics.
    return dict(safe_to_continue=not failures,failures=failures,comparisons=comparisons,
                absolute_quality='NOT_ACCEPTED_WITHOUT_ACTION_AND_VISUAL_REVIEW')


def compare_geometry_quality(rows,baseline):
    """Additional retention checks; never replace the recorded v3 gate."""
    current,reference=groups(rows),groups(baseline)
    failures=[]
    comparisons=[]
    if not rows or set(current)!=set(reference):failures.append('geometry panel groups differ/empty')
    metrics=['geometry_sole_speed_m_s','geometry_near_floor_fraction','joint_limit_max_deg','joint_limit_frame_fraction']
    for key,old in reference.items():
        new=current.get(key,[])
        if sorted((r['sample'],r['seed']) for r in new)!=sorted((r['sample'],r['seed']) for r in old):
            failures.append(f'{key}: geometry pairs differ')
            continue
        for metric in metrics:
            a=[r.get(metric) for r in new];b=[r.get(metric) for r in old]
            if any(v is None or not math.isfinite(v) for v in a+b):
                failures.append(f'{key}: invalid {metric}');continue
            # Worst seed for limit maxima; mean for continuous error proxies.
            x,y=(max(a),max(b)) if metric=='joint_limit_max_deg' else (sum(a)/len(a),sum(b)/len(b))
            comparisons.append(dict(group=list(key),metric=metric,current=x,baseline=y))
            if metric=='geometry_sole_speed_m_s' and x>max(.03,1.25*y):failures.append(f'{key}: sole speed regression')
            if metric=='geometry_near_floor_fraction' and x<.8*y:failures.append(f'{key}: near-floor coverage loss')
            if metric=='joint_limit_max_deg' and x>max(2.,1.1*y):failures.append(f'{key}: joint-limit regression {x:.3f}>{y:.3f}')
            if metric=='joint_limit_frame_fraction' and x>max(.02,1.25*y):failures.append(f'{key}: limit-violation frequency regression')
    return dict(safe_to_continue=not failures,failures=failures,comparisons=comparisons,
        absolute_quality='NOT_ACCEPTED_WITHOUT_ACTION_AND_VISUAL_REVIEW')
