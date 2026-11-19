"""Portable interaction annotations, independent of rendering and inference.

Raw input is an editing log. Accepted controls describe the final motion timeline.
Keep both: a click's reception frame is not its target deadline or adoption frame.
"""
from copy import deepcopy
import json
from pathlib import Path
import threading
import time
import uuid


def capture_id(session):
    model = getattr(session, 'model', None)
    if getattr(model, 'sid', None) is not None:
        return model.sid
    return session.__dict__.setdefault('interaction_session_id', uuid.uuid4().hex)


def reset_local_capture(session):
    session.interaction_session_id = uuid.uuid4().hex
    session.interaction_inputs = []
    session.interaction_local_events = []
    session.interaction_loaded_replay = None
    session.interaction_loaded_provenance = None
    session.interaction_loaded_constraints = None
    session.interaction_freeze_loaded_constraints = False


def restore_interaction_export(session, exported):
    """Keep annotations portable when a saved motion is loaded without inference."""
    reset_local_capture(session)
    session.interaction_loaded_replay = deepcopy(exported.get('interaction_replay'))
    session.interaction_loaded_provenance = deepcopy(exported.get('provenance'))
    session.interaction_loaded_constraints = deepcopy(exported.get('constraints'))
    session.interaction_freeze_loaded_constraints = True


def local_condition_snapshot(session, motion_mask, observed_motion, condition_start):
    """Copy the exact conditioning argument used by the official ARDY call."""
    g = session.gui_elements
    control = dict(root2d=[], velocity_enabled=bool(g.gui_use_target_velocity_checkbox.value),
                   velocity=list(g.gui_target_root_velocity.value),
                   heading_enabled=bool(g.gui_use_target_heading_checkbox.value))
    if motion_mask is None or observed_motion is None:
        control['observed_argument'] = None
        return control
    mask = motion_mask.detach().cpu()
    observed = observed_motion.detach().cpu()
    if mask.ndim == 2:
        mask, observed = mask[None], observed[None]
    rows = []
    for sample_mask, sample_observed in zip(mask, observed):
        indices = sample_mask.bool().nonzero(as_tuple=False)
        values = sample_observed[indices[:, 0], indices[:, 1]]
        rows.append(dict(indices=indices.tolist(), values=values.tolist()))
    control['observed_argument'] = dict(shape=list(mask.shape), samples=rows,
        frame_origin=int(condition_start), representation='exact observed_motion argument to autoregressive_step')
    root = session.constraints.get('2D Root')
    if root is not None:
        for frame, position in sorted(root.keyframes.items()):
            offset = int(frame)-int(condition_start)
            if not 0 <= offset < mask.shape[1] or not bool(mask[0, offset].any()):
                continue
            pos = position.detach().cpu().tolist() if hasattr(position, 'detach') else position.tolist() if hasattr(position, 'tolist') else list(position)
            ids = list(root.frame2keyid.get(frame, []))
            point = dict(frame=offset, xz=[float(pos[0]),float(pos[2])], keyframe_ids=ids,
                         origin='keyboard_preview' if ids and all(str(k).startswith('target_vel_waypoint_') for k in ids) else 'spatial_constraint')
            if frame in root.root_headings:
                point['heading'] = float(root.root_headings[frame])
            control['root2d'].append(point)
    return control


def record_local_acceptance(session, start, condition_start, control):
    """Commit annotations in the same motion lock as the generated trajectory."""
    lock = session.__dict__.setdefault('interaction_record_lock', threading.RLock())
    record = dict(event='control_accepted',session=capture_id(session),source_start=int(start),
                  condition_source_start=int(condition_start),control=deepcopy(control),
                  wall_time=time.time(),monotonic=time.monotonic())
    with lock:
        session.__dict__.setdefault('interaction_local_events', []).append(record)


def source_state(session):
    model = getattr(session, 'model', None)
    return dict(source_frame=int(getattr(session, 'frame_idx', -1)),
                consumed_source_frame=float(getattr(model, 'consumed_frame', getattr(session,'frame_idx',0))),
                planning_frame=(int(model.planning_frame) if getattr(model, 'planning', False) else None),
                playing=bool(getattr(session, 'playing', False)))


def record_input(session, event, **fields):
    """Copy small human-input events; never run inference or save pose tensors here."""
    lock = session.__dict__.setdefault('interaction_record_lock', threading.RLock())
    session.interaction_freeze_loaded_constraints = False
    model = getattr(session, 'model', None)
    record = dict(event=event, wall_time=time.time(), monotonic=time.monotonic(),
                  session=capture_id(session), **source_state(session), **fields)
    with lock:
        session.__dict__.setdefault('interaction_inputs', []).append(deepcopy(record))
    if hasattr(model, 'event'):
        model.event('interaction_input', input=record)


def build_interaction_export(session, frames):
    """Export the final adopted control schedule and all input/acceptance events.

    Replanning replaces a tail; stale accepted conditions must not appear on the
    final replay. Root point offsets are relative to the 52-frame model window,
    including when a replan is adopted in the middle of that window.
    """
    model = getattr(session, 'model', None)
    sid = capture_id(session)
    run_dir = getattr(model, 'run_dir', None)
    events, warnings = [], []
    if run_dir is not None:
        path = Path(run_dir) / 'events.jsonl'
        if path.exists():
            for line in path.read_text().splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    warnings.append('Incomplete diagnostic event was skipped.')
                    continue
                if event.get('session') == sid:
                    events.append(event)
    lock = session.__dict__.setdefault('interaction_record_lock', threading.RLock())
    with lock:
        local_events = deepcopy(getattr(session,'interaction_local_events',[]))
        loaded = deepcopy(getattr(session,'interaction_loaded_replay',None))
        if loaded and not local_events and not getattr(session,'interaction_inputs',[]) and int(loaded['frames'])==int(frames):
            return loaded
    if loaded:
        # Frozen annotations describe the imported motion, not newly generated poses.
        events[:0] = [dict(event='control_accepted',source_start=e['start_frame'],
            condition_source_start=e['condition_source_start'],control=e['control'],
            monotonic=e.get('accepted_monotonic'),request_version=e.get('request_version'),
            revision=e.get('revision')) for e in loaded.get('accepted_control_timeline',[])]
    events.extend(local_events)
    timeline = []
    for event in events:
        if event.get('event') != 'control_accepted':
            continue
        start = int(event['source_start'])
        # Replacement events beyond a later trimmed motion are retained in the
        # editing log, but cannot define the exported motion's control schedule.
        timeline = [entry for entry in timeline if entry['start_frame'] < start]
        if start >= frames:
            continue
        control = deepcopy(event['control'])
        base = int(event.get('condition_source_start', start // 52 * 52))
        points = [dict(point, frame=base+int(point['frame'])) for point in control.get('root2d', [])]
        timeline.append(dict(start_frame=start, condition_source_start=base,
                             control=control, root2d_absolute=points,
                             accepted_monotonic=event.get('monotonic'),
                             request_version=event.get('request_version'), revision=event.get('revision')))
    for i,entry in enumerate(timeline):
        entry['end_frame_exclusive'] = timeline[i+1]['start_frame'] if i+1<len(timeline) else frames
    lock = session.__dict__.setdefault('interaction_record_lock', threading.RLock())
    with lock:
        inputs = deepcopy([e for e in getattr(session, 'interaction_inputs', [])
                           if e.get('session') in (None, sid)])
    # Logs allow re-export/recovery when an input is no longer in live memory.
    known = {(e['monotonic'],e['event']) for e in inputs}
    for event in events:
        item = event.get('input') if event.get('event') == 'interaction_input' else None
        if item and (item['monotonic'],item['event']) not in known:
            inputs.append(item)
            known.add((item['monotonic'],item['event']))
    if loaded:
        inputs = deepcopy(loaded.get('input_events',[])) + inputs
    inputs.sort(key=lambda e:e['monotonic'])
    expected = [int(entry[0]) for entry in getattr(getattr(model,'control_history',None),'_entries',())
                if int(entry[0])<frames]
    remote = bool(getattr(model,'is_remote_mm2',False))
    complete = bool(timeline and timeline[0]['start_frame']==0 and
                    (not remote or [e['start_frame'] for e in timeline]==expected))
    if frames and not complete:
        warnings.append('Accepted control coverage is incomplete; do not infer missing directions.')
    return dict(version='interaction_replay_v1', fps=float(session.model_fps),
                frames=int(frames), source_session=sid,
                coordinate_system='ARDY world: Y-up, XZ ground plane, metres; heading=atan2(vx,vz)',
                ardy_to_mujoco_matrix=[[0,0,1],[1,0,0],[0,1,0]],
                interval_convention='start inclusive, end_frame_exclusive exclusive; point frames are absolute',
                input_events=inputs, accepted_control_timeline=timeline,
                generation_events=[e for e in events if e.get('event') in (
                    'control_received','control_accepted','replacement_rejected','replacement_deferred',
                    'reference_replaced','published','playback_seek','planning_selected')],
                coverage=dict(accepted_controls_complete=complete,
                              raw_input_capture=bool(getattr(session,'interaction_capture_enabled',False)),
                              physics_actual_included=False),
                source_log_directory=str(run_dir) if run_dir is not None else None,
                warnings=warnings)
