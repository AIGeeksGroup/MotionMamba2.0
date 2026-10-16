"""Strict frozen adapter around pinned OMG G1 motion/text encoders."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from main3_training import file_hash
from g1_evaluator_conversion import resample_qpos

SOURCE = Path('/data3/zts/packages/OMG-1f2aebd58532631c48b9a09a1580d2bf2c5afd14')
ASSETS = Path('/data3/zts/main3_20260910/evaluator')
sys.path.insert(0, str(SOURCE/'src'))


class G1LearnedEvaluator:
    def __init__(self, device='cpu'):
        from omg.benchmarks.evaluator.motion_encoder import MotionEncoder
        from omg.robots.g1.kinematics import G1Kinematics
        self.device = torch.device(device)
        path = ASSETS/'evaluator/step_004000.pt'
        if file_hash(path) != 'e346790c5ce33c42840aa19f7fcc2891a4cab27e0d92e23bfaff7d48cb339c40':
            raise ValueError('OMG weight identity changed')
        self.checkpoint = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        cfg = self.checkpoint['config']
        self.kinematics = G1Kinematics(SOURCE/'assets/robots/g1/g1_kinematics.json').to(self.device)
        if cfg['motion_key'] != 'body_pos_local' or cfg['fps'] != 30 or cfg['segment_length'] != 60:
            raise ValueError('unreviewed evaluator representation or sampling contract')
        kwargs = {key: cfg[key] for key in ['movement_dim','hidden_dim','movement_mode','temporal_kind',
                                           'num_layers','num_heads','mlp_ratio','dropout']}
        self.motion_encoder = MotionEncoder(input_dim=self.kinematics.num_bodies*3,
                                             output_dim=cfg['embedding_dim'], **kwargs).to(self.device)
        self.motion_encoder.load_state_dict(self.checkpoint['motion_encoder'], strict=True)
        self.motion_encoder.eval().requires_grad_(False)
        self.text_encoder = None
        self.contract = dict(encoder='OMG G1 evaluator', weight_sha256=file_hash(path),
                             source_revision=SOURCE.name.removeprefix('OMG-'), motion_key=cfg['motion_key'],
                             input_dim=90, embedding_dim=512, fps=30, trained_frames=60,
                             normalization='official first-frame root translation and inverse heading; no external stats',
                             embedding_normalized=True, text='google-t5/t5-3b with released text projection',
                             text_max_length=cfg['text_max_length'], precision='FP32',
                             kinematics_sha256=file_hash(SOURCE/'assets/robots/g1/g1_kinematics.json'),
                             resampling='25->30 physical timestamps; linear roots, unwrapped hinges, quaternion SLERP; floor last timestamp',
                             ragged_batch='encode each valid motion independently; no padded convolution contamination',
                             retrieval='motion-to-text, squared Euclidean, fixed 32-caption pools, stable ID-order tie break',
                             long_windows='60 frames, stride 60, final 60-frame overlap covers remainder',
                             evaluator_training_includes_bones_seed=True,
                             exact_training_test_motion_overlap='unknown; evaluator training split IDs unavailable')

    @torch.inference_mode()
    def features(self, qpos, fps=25):
        from omg.benchmarks.evaluator.representation import canonical_body_positions_from_qpos
        values = resample_qpos(qpos, fps, 30)[0] if fps != 30 else qpos
        return canonical_body_positions_from_qpos(torch.tensor(values, device=self.device)[None], self.kinematics)

    @torch.inference_mode()
    def encode_motion(self, qpos, fps=25):
        x = self.features(qpos, fps)
        valid = torch.ones(x.shape[:2], device=self.device, dtype=torch.bool)
        return self.motion_encoder(x, valid_mask=valid)[0].cpu().numpy()

    @torch.inference_mode()
    def encode_windows(self, qpos):
        values, times = resample_qpos(qpos)
        if len(values) < 60:
            raise ValueError('long sequence has no legal evaluator window')
        starts = list(range(0, len(values)-59, 60))
        if starts[-1] != len(values)-60:
            starts.append(len(values)-60)
        return np.stack([self.encode_motion(values[s:s+60], fps=30) for s in starts]), starts

    def load_text(self):
        from omg.benchmarks.evaluator.text_encoder import TextEncoder
        if self.text_encoder is None:
            self.text_encoder = TextEncoder(output_dim=512, model_name=str(ASSETS/'t5-3b'),
                                             max_length=self.contract['text_max_length']).to(self.device)
            self.text_encoder.proj.load_state_dict(self.checkpoint['text_proj_state_dict'], strict=True)
            self.text_encoder.eval().requires_grad_(False)
        return self.text_encoder

    @torch.inference_mode()
    def encode_texts(self, captions, batch_size=4):
        encoder = self.load_text()
        return np.concatenate([encoder(captions[i:i+batch_size], self.device).cpu().numpy()
                               for i in range(0, len(captions), batch_size)])


def frechet(real, generated):
    """Full population covariance FID using a symmetric PSD square root."""
    a, b = np.asarray(real, dtype=np.float64), np.asarray(generated, dtype=np.float64)
    if min(len(a),len(b)) < 2 or a.shape[1] != b.shape[1] or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('invalid population embeddings for FID')
    ca, cb = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    vals, vecs = np.linalg.eigh(ca)
    if vals.min() < -1e-8:
        raise ValueError('non-PSD real covariance')
    root = (vecs*np.sqrt(np.maximum(vals, 0))) @ vecs.T
    middle = root @ cb @ root
    eigen = np.linalg.eigvalsh((middle+middle.T)*0.5)
    if eigen.min() < -1e-8:
        raise ValueError('non-PSD covariance product')
    raw = float(np.square(a.mean(0)-b.mean(0)).sum()+np.trace(ca)+np.trace(cb)-2*np.sqrt(np.maximum(eigen,0)).sum())
    if raw < -1e-6:
        raise ValueError(f'significantly negative FID: {raw}')
    return dict(fid=max(raw,0), raw_fid=raw, numerical_correction='tiny negative eigenvalues clipped to zero',
                real_samples=len(a), generated_samples=len(b), covariance_dim=a.shape[1])


def retrieval_pools(rows, seed=20260910):
    rng = np.random.default_rng(seed)
    pools = []
    for i, row in enumerate(rows):
        candidates, seen = [], {row['caption']}
        for j in rng.permutation(len(rows)):
            other = rows[int(j)]
            if other['mother_id'] == row['mother_id'] or other['caption'] in seen:
                continue
            seen.add(other['caption'])
            candidates.append(int(j))
            if len(candidates) == 31:
                break
        if len(candidates) != 31:
            raise ValueError('not enough unique captions for 32-candidate retrieval')
        pools.append(sorted([i]+candidates, key=lambda j: rows[j]['id']))
    return pools


def retrieval(motion, text, pools):
    ranks = []
    for i, pool in enumerate(pools):
        distances = np.square(motion[i].astype(np.float64)-text[pool].astype(np.float64)).sum(-1)
        order = np.argsort(distances, kind='stable')
        ranks.append(int(np.flatnonzero(np.asarray(pool)[order] == i)[0])+1)
    return dict(top1=float(np.mean(np.array(ranks)<=1)), top2=float(np.mean(np.array(ranks)<=2)),
                top3=float(np.mean(np.array(ranks)<=3)), ranks=ranks, pool_size=32,
                positive_sets=[[i] for i in range(len(ranks))])
