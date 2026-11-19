"""Clean ground-truth context for single-window diffusion supervision."""
import torch

from . import batching
from .checkpoint import tensor_hash
from motion_mamba2.model.history_commit import canonical_history_prime


def _commit_history(model, client, fixed, records, audit=False):
    raw = model.denoiser.model
    frames = records[0]['H']
    tokens = frames // raw.num_frames_per_token
    summary = dict(history_frames=frames, history_source='clean_ground_truth',
        predicted_history_frames=0, generated_rollout_windows=0,
        accepted_history_commits=0, cross_window_truncated_bptt=False,
        future_GT_committed=False)
    if not frames:
        return fixed, summary

    # Encode the sampled clean prefix once. This is context preparation, not
    # a diffusion prediction of history or an autoregressive training rollout.
    # Keep its parameter graph so the continuation loss trains PCM writes too.
    history = fixed['target'][:, :tokens].detach()
    if audit:
        before = {key: tensor_hash(fixed[key]) for key in
                  ['target', 'noise', 'x', 'supervised_motion', 'observed', 'motion_mask']}
        assert torch.equal(history, fixed['x'][:, :tokens])
        assert torch.equal(fixed['decoder_history_hybrid'], history[:, -13:])
    canonical_history_prime(raw, history, text_feat=fixed['text'],
        text_pad_mask=fixed['text_mask'], first_heading_angle=fixed['heading'],
        next_generation_frames=52, chunk_id=0, detach=False,
        include_synthetic_generation=False)
    controller = client.controller
    controller._completed_windows = controller._accepted_commits = controller._next_chunk_id = 1
    controller._last_text_feat = fixed['text']
    controller._last_text_pad_mask = fixed['text_mask']
    if audit:
        assert all(tensor_hash(fixed[key]) == value for key, value in before.items())
        summary.update(history_hash=tensor_hash(history),
            GT_B_target_hash=tensor_hash(fixed['target'][:, tokens:tokens + 13]),
            G_noise_hash=tensor_hash(fixed['noise'][:, tokens:tokens + 13]),
            G_input_hash=tensor_hash(fixed['x'][:, tokens:tokens + 13]),
            clean_history_matches_input=True, GT_continuation_unchanged=True)

    # History is already represented in the committed states. Feed only the
    # noisy generation window and the visible future controls to prediction.
    for key in ['x', 'target', 'noise', 'history_token_mask',
                'generation_token_mask', 'future_token_mask']:
        fixed[key] = fixed[key][:, tokens:]
    for key in ['history_mask', 'generation_mask', 'future_mask',
                'observed', 'motion_mask', 'pad_mask']:
        fixed[key] = fixed[key][:, frames:]
    fixed['history_len'] = torch.zeros_like(fixed['history_len'])
    fixed['lengths'] = fixed['lengths'] - frames
    fixed.update(generated_history_state_ready=True, sequential_cache=True)
    summary.update(accepted_history_commits=1, differentiable_GT_context=True)
    return fixed, summary


def build(model, client, prepared, config, audit=False):
    fixed, receipts = batching.build_group(model, client, prepared, config,
                                            audit=audit, serial_codec=True)
    records = [item['record'] for item in prepared]
    fixed, summary = _commit_history(model, client, fixed, records, audit)
    for receipt in receipts:
        receipt.update(accepted_history_commits=summary['accepted_history_commits'],
            history_source='clean_ground_truth', predicted_history_frames=0,
            cross_window_truncated_bptt=False, explicit_history_frames=0,
            duplicated_frames=0, future_GT_committed=False)
    return fixed, receipts, summary
