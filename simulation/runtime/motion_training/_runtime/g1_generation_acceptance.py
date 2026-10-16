"""Predeclared engineering checks, separate from loss and relative retention.

Passing numeric checks is necessary, not sufficient: action/visual review and
an untouched confirmation panel are required before a Native is accepted.
"""
import math

POLICY = {
    'version': 'g1_raw_generation_v1_20260920',
    'contact_speed_m_s_max': .15,
    'geometry_sole_speed_m_s_max': .40,
    'contact_fraction_min': .20,
    'geometry_near_floor_fraction_min': .20,
    'joint_limit_max_deg_max': 5.,
    'joint_limit_frame_fraction_max': .01,
    'jitter_m_per_frame2_max': .03,
    'position_rotation_disagreement_cm_max': 3.,
    'movement_speed_ranges': {'walk': [.15, 1.6], 'slow_walk': [.03, .8],
                              'side_walk': [.08, 1.3], 'turn': [.08, 1.6]},
    'stationary_speed_max': .15,
    'wave_arm_range_deg_min': 10.,
    'squat_knee_range_deg_min': 15.,
    'squat_root_height_range_m_min': .08,
    'note': 'Engineering acceptance targets, not official benchmark thresholds or evidence of semantic correctness.',
}


def check_generation(rows, expected_cases, policy=POLICY):
    expected = {(r['id'], r['condition'], r['frames'], r['seed']) for r in expected_cases}
    actual = [(r['sample'], r['condition'], r['frames'], r['seed']) for r in rows]
    failures = []
    if not rows or len(actual) != len(set(actual)) or set(actual) != expected:
        failures.append({'case': None, 'reason': 'Missing, duplicate or mismatched cases'})
    for row in rows:
        case = [row['sample'], row['condition'], row['frames'], row['seed']]
        checks = {key[:-4]: (None, value) for key, value in policy.items()
                  if key.endswith('_max') and key != 'stationary_speed_max'}
        checks.update({key[:-4]: (value, None) for key, value in policy.items()
                       if key.endswith('_min') and not key.startswith(('wave_', 'squat_'))})
        category = row['category']
        if category in policy['movement_speed_ranges']:
            checks['root_speed_m_s'] = policy['movement_speed_ranges'][category]
        else:
            checks['root_speed_m_s'] = (0., policy['stationary_speed_max'])
        if category == 'wave':
            checks['arm_range_deg'] = (policy['wave_arm_range_deg_min'], None)
        if category == 'squat':
            checks['knee_range_deg'] = (policy['squat_knee_range_deg_min'], None)
            checks['root_height_range_m'] = (policy['squat_root_height_range_m_min'], None)
        for key, (low, high) in checks.items():
            value = row.get(key)
            if value is None or not math.isfinite(value) or (low is not None and value < low) or (high is not None and value > high):
                failures.append(dict(case=case, metric=key, value=value, lower=low, upper=high))
    return dict(numeric_pass=not failures, failures=failures, policy=policy,
                status='NUMERIC_PASS_REQUIRES_CONFIRMATION_AND_VISUAL_REVIEW' if not failures else 'NOT_ACCEPTED',
                accepted=False)
