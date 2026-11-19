"""Family-isolated, deterministic continuous-GT crops; no teacher labels."""
import collections
import hashlib
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch

BUCKETS = [(0, 0), (52, 0), (196, 0), (0, 196), (52, 144), (144, 52)]
CONDITIONS = ['none', 'root_trajectory', 'root_waypoint', 'full_keyframe',
              'joint_position', 'joint_rotation']
IDENTITIES = ['source_family_id', 'source_motion_id', 'sample_id', 'feature_path']


def text_key(text):
    return hashlib.sha256(text.encode()).hexdigest()


def validate_record(record):
    if record['supervision'] != 'GT' or (record['H'], record['F']) not in BUCKETS:
        raise ValueError('Expected a real-GT H/G/F crop')
    if record['G'] != 52 or record['crop_end'] - record['crop_start'] != record['H'] + 52 + record['F']:
        raise ValueError('Invalid GT crop length')
    if record['condition'] not in CONDITIONS or not record['caption'].strip():
        raise ValueError('Invalid text or spatial control')
    return record


def intervals(record, frames):
    if record['frames'] < frames:
        return []
    if not record.get('caption_time_spans'):
        captions = record.get('captions') or [record.get('caption', '')]
        return [(0, record['frames'] - frames, text) for text in captions if text.strip()]
    result = []
    origin = record.get('native_start_sec', 0)
    for span in record['caption_time_spans']:
        low = max(0, math.ceil((span['native_start_sec'] - origin) * 25 - 1e-6))
        high = min(record['frames'] - frames,
                   math.ceil((span['native_end_sec'] - origin) * 25 - 1e-6) - frames)
        if low <= high and span['text'].strip():
            result.append((low, high, span['text']))
    return result


class GTPlan:
    def __init__(self, config):
        self.seed = config['seed']
        options = config['data']
        index = Path(options['index_path']).expanduser().resolve()
        if not options['index_path'] or not index.is_file():
            raise ValueError('Fill data.index_path in the training configuration')
        root = Path(options['feature_root']).expanduser().resolve() if options['feature_root'] else index.parent
        records = [json.loads(line) for line in index.read_text().splitlines() if line.strip()]
        heldout = {str(row[key]) for row in records if str(row.get('project_split', '')).lower() != 'train'
                   for key in IDENTITIES if row.get(key)}
        heldout.update(str(value) for value in options['excluded_families'])
        self.groups = {}
        self.feature_paths = set()
        self.captions = set()
        for source in ['bones', 'humanml3d', 'hiphi']:
            for history, future in BUCKETS:
                families = collections.defaultdict(list)
                for row in records:
                    if row['dataset'] != source or row.get('project_split') != 'train':
                        continue
                    if any(str(row.get(key, '')) in heldout for key in IDENTITIES):
                        continue
                    choices = intervals(row, history + 52 + future)
                    if not choices:
                        continue
                    row = dict(row)
                    feature = Path(row['feature_path'])
                    row['feature_path'] = str(feature if feature.is_absolute() else root / feature)
                    if not Path(row['feature_path']).is_file():
                        raise FileNotFoundError(row['feature_path'])
                    families[row['source_family_id']].append((row, choices))
                    self.feature_paths.add(row['feature_path'])
                    self.captions.update(choice[2] for choice in choices)
                if not families:
                    raise ValueError(f'No isolated training families for {source}, H={history}, F={future}')
                self.groups[source, history, future] = (sorted(families), families)
        # A caption cache is a frozen encoder output, never an action teacher.
        self.text_files = {}
        roots = [Path(path).expanduser().resolve() for path in options['text_features_paths']]
        for caption in sorted(self.captions):
            path = next((root / (text_key(caption) + '.npy') for root in roots
                         if (root / (text_key(caption) + '.npy')).is_file()), None)
            if path is None:
                raise FileNotFoundError(f'Missing cached text feature: {caption}')
            self.text_files[caption] = str(path)

    def records_at(self, step, count=512):
        if count != 512:
            raise ValueError('This recipe always uses global batch 512')
        layouts = [layout for layout, repeat in zip(BUCKETS, [2, 5, 4, 2, 4, 3]) for _ in range(repeat)]
        np.random.default_rng(self.seed + 1000003 * (step // 20)).shuffle(layouts)
        history, future = layouts[step % 20]
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, step, history, future]))

        def cycle(values):
            result = []
            while len(result) < count:
                part = list(values)
                rng.shuffle(part)
                result.extend(part)
            return result[:count]

        sources = cycle(['bones'] * 14 + ['humanml3d'] * 3 + ['hiphi'] * 3)
        conditions = cycle(['none'] * 5 + [value for value in CONDITIONS[1:] for _ in range(4)])
        result = []
        nonempty = 0
        for lane, (source, condition) in enumerate(zip(sources, conditions)):
            keys, families = self.groups[source, history, future]
            family = keys[int(rng.integers(len(keys)))]
            rows = families[family]
            row, choices = rows[int(rng.integers(len(rows)))]
            low, high, caption = choices[int(rng.integers(len(choices)))]
            start = int(rng.integers(low, high + 1))
            out = False
            if future and condition != 'none':
                out = nonempty % 2 == 0
                nonempty += 1
            slot = step * 512 + lane
            result.append(validate_record(dict(id=f'gt:{slot}', dataset=source,
                source_family_id=family, sample_id=row['sample_id'], feature_path=row['feature_path'],
                crop_start=start, crop_end=start + history + 52 + future,
                caption=caption, H=history, G=52, F=future, supervision='GT',
                condition=condition, out_of_horizon=out, seed=self.seed + 1000 + slot)))
        return result


def sample_control_condition(
    features: torch.Tensor,
    skeleton,
    rng: Optional[np.random.Generator] = None,
    horizon: int = 40,
    control: Optional[str] = None,
    out_of_horizon: Optional[bool] = None,
    condition_dropout: float = 0.0,
    generation_start: int = 0,
) -> dict[str, torch.Tensor | str | int | bool]:
    """Sample visible spatial constraints from the real GT crop."""
    if features.ndim != 2 or (features.shape[-1] - 6) % 12:
        raise ValueError(f"expected a [T,12J+6] feature tensor, got {tuple(features.shape)}")
    joints = (features.shape[-1] - 6) // 12
    rng = rng or np.random.default_rng()
    t = features.shape[0]
    controls = ("root_trajectory", "root_waypoint", "full_keyframe", "joint_position", "joint_rotation")
    control = control or controls[int(rng.integers(len(controls)))]
    if control not in controls:
        raise ValueError(f"unknown control {control!r}")
    use_out = bool(rng.integers(2)) if out_of_horizon is None else bool(out_of_horizon)
    generation_end = generation_start + horizon
    use_out = use_out and t > generation_end + 1
    if use_out:
        low = generation_end + 1
        frame = int(rng.integers(low, t))
    else:
        low = min(generation_start, t - 1)
        high = min(generation_end, t)
        frame = int(rng.integers(low, max(low + 1, high)))
    observed = torch.zeros_like(features)
    mask = torch.zeros_like(features, dtype=torch.bool)
    root = slice(0, 3)
    heading = slice(3, 5)
    local_end = 5 + (joints - 1) * 3
    rotations_end = local_end + joints * 6
    local = slice(5, local_end)
    rotations = slice(local_end, rotations_end)
    if control == "root_trajectory":
        start = generation_start if not use_out else max(generation_end, frame - horizon)
        mask[start:frame + 1, 0] = True
        mask[start:frame + 1, 2] = True
    elif control == "root_waypoint":
        mask[frame, 0] = True
        mask[frame, 2] = True
    elif control == "full_keyframe":
        mask[frame, root] = True
        mask[frame, heading] = True
        mask[frame, local] = True
        mask[frame, rotations] = True
    elif control == "joint_position":
        joint = int(rng.integers(1, joints))
        # End-effector position constraints carry root xyz and heading so
        # the relative joint position has an unambiguous world-space frame.
        mask[frame, 0:3] = True
        mask[frame, heading] = True
        mask[frame, 5 + (joint - 1) * 3:5 + joint * 3] = True
    else:
        joint = int(rng.integers(0, joints))
        mask[frame, local_end + joint * 6:local_end + (joint + 1) * 6] = True
    observed[mask] = features[mask]
    dropped = bool(rng.random() < condition_dropout)
    if dropped:
        observed.zero_()
        mask.zero_()
    return {
        "observed_motion": observed,
        "motion_mask": mask,
        "control": control,
        "frame": frame,
        "out_of_horizon": use_out,
        "condition_dropped": dropped,
    }
