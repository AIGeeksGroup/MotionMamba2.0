"""Versioned frozen-codec and decoder-history contract for corrected G1 runs."""
import hashlib
import json
from pathlib import Path

import torch
from contextlib import contextmanager

CONTRACT = 'g1_official_codec_v2'
GENERATION_CONTRACT = 'g1_committed_v3'


def advance_validation_guard(state, value, step, ratio, patience=2):
    """A fixed validation panel must not regress persistently versus its best."""
    import math
    state = dict(state)
    previous_best = state.get('best', float('inf'))
    improved = math.isfinite(value) and value < previous_best
    if improved:
        state.update(best=value, best_step=step, bad_count=0)
    elif not math.isfinite(value) or value > previous_best*ratio:
        state['bad_count'] = state.get('bad_count', 0)+1
    else:
        state['bad_count'] = 0
    state.update(step=step, value=value, stop=state.get('bad_count', 0)>=patience)
    return state, improved


def stats_hash(directory):
    h = hashlib.sha256()
    for name in ('mean.npy', 'std.npy'):
        h.update(name.encode())
        h.update((Path(directory)/name).read_bytes())
    return h.hexdigest()


def inspect_stats(directory):
    path = Path(directory)/'motion_contract.json'
    if not path.exists():
        return {'version': 'legacy'}
    metadata = json.loads(path.read_text())
    if metadata['version'] != CONTRACT or metadata['stats_sha256'] != stats_hash(directory):
        raise ValueError('Unknown or modified motion statistics contract')
    return metadata


def validate_codec_stats(official_rep, replacement, metadata):
    if metadata['version'] == 'legacy':
        return
    for field in ('body_stats', 'local_root_stats'):
        for stat in ('mean', 'std'):
            if not torch.equal(getattr(getattr(official_rep, field), stat).cpu(),
                               getattr(getattr(replacement, field), stat).cpu()):
                raise ValueError(f'Frozen official codec statistics changed: {field}.{stat}')


def install_decoder_history(model, frames=52):
    """Keep decoder history independent from the denoiser's explicit-history crop."""
    if frames <= 0 or frames % model.hybrid.num_frames_per_token:
        raise ValueError('Decoder history must be a positive patch-aligned length')
    if getattr(model, 'decoder_history_frames', None) is not None:
        if model.decoder_history_frames != frames:
            raise ValueError('Decoder history contract already installed differently')
        return
    original = model.hybrid.get_explicit_motion_from_hybrid_autoregressive

    def decode(*args, **kwargs):
        kwargs['crop_history_length'] = frames
        return original(*args, **kwargs)

    model.hybrid.get_explicit_motion_from_hybrid_autoregressive = decode
    model.decoder_history_frames = frames


def validate_checkpoint(checkpoint, metadata):
    config = checkpoint.get('config', checkpoint.get('metrics', {}).get('config', {}))
    saved = checkpoint.get('motion_contract', config.get('motion_contract', 'legacy'))
    if saved != metadata['version']:
        raise ValueError(f'Checkpoint/representation mismatch: {saved} != {metadata["version"]}')
    if saved == CONTRACT:
        fingerprint = checkpoint.get('stats_sha256', config.get('stats_sha256'))
        if fingerprint != metadata['stats_sha256']:
            raise ValueError('Checkpoint normalization fingerprint mismatch')


def decode_training_prediction(prediction, fixed):
    """Decode only real history/generation, with a separate accepted decoder prefix."""
    converter = fixed['hybrid_converter']
    if fixed.get('motion_contract', 'legacy') != CONTRACT:
        return converter.get_explicit_motion_from_hybrid(prediction, fixed['pad_mask'], fixed['lengths'])
    valid = fixed['history_mask'] | fixed['generation_mask']
    prefix = fixed.get('decoder_history_hybrid')
    count = 0
    if prefix is not None:
        count = prefix.shape[1]*converter.num_frames_per_token
        prediction = torch.cat((prefix.to(prediction), prediction), dim=1)
        valid = torch.cat((torch.ones(valid.shape[0], count, device=valid.device, dtype=torch.bool), valid), dim=1)
    decoded = converter.get_explicit_motion_from_hybrid(prediction, valid, valid.sum(1))
    return decoded[:, count:]


def decode_committed_window(model, world, previous=None, *, valid_frames=None):
    """Decode an accepted window once, retaining past latent context in world space."""
    patch = model.hybrid.num_frames_per_token
    requested = world.shape[1] * patch if valid_frames is None else int(valid_frames)
    if not 0 < requested <= world.shape[1] * patch:
        raise ValueError('valid_frames must fit the committed window')
    # The codec operates on whole patches; only the last partial patch can
    # extend past the requested output. Never decode additional full tokens.
    world = world[:, :(requested + patch - 1)//patch]
    context = world if previous is None else torch.cat((previous.to(world), world), dim=1)
    frames = context.shape[1]*model.hybrid.num_frames_per_token
    pad = torch.ones(context.shape[0], frames, device=context.device, dtype=torch.bool)
    decoded = model.hybrid.get_explicit_motion_from_hybrid(context, pad, pad.sum(1))
    tokens = model.decoder_history_frames//model.hybrid.num_frames_per_token
    return decoded[:, -world.shape[1]*patch:][:, :requested], context[:, -tokens:].detach().clone()


@contextmanager
def capture_committed_decode(model, frames, previous=None):
    """One batched decode lifecycle for inference, evaluation and training rollout.

    The caller primes denoiser state separately and supplies the same encoded
    physical prefix to this decoder. Outputs are immutable accepted windows.
    """
    from ardy.model.ardy_model import translate_normalized_root_motion
    original = model._sequence_kernel_post_window_hook
    outputs = []
    translation = None
    remaining = int(frames)

    def committed(sequence, center, root, heading, window):
        nonlocal previous, translation, remaining
        original(sequence, center, root, heading, window)
        translation = center.clone() if translation is None else translation + center
        accepted = sequence[:, -model.gen_horizon_len//model.hybrid.num_frames_per_token:]
        local_root, body = model.hybrid.get_root_and_latent_body_motion_from_hybrid(accepted)
        world_root = translate_normalized_root_motion(local_root, translation, model.motion_rep)
        world = model.hybrid.get_hybrid_motion_from_root_and_latent_body_motion(world_root, body)
        count = min(remaining, model.gen_horizon_len)
        explicit, previous = decode_committed_window(model, world, previous, valid_frames=count)
        outputs.append(explicit)
        remaining -= count

    model._sequence_kernel_post_window_hook = committed
    try:
        yield outputs
        if remaining != 0:
            raise RuntimeError(f'Incomplete committed decoding: {remaining} frames remain')
    finally:
        model._sequence_kernel_post_window_hook = original


def generate_with_committed_decode(model, texts, frames, *, decoder_prefix=None, **kwargs):
    if getattr(model, 'generation_contract', 'legacy') != GENERATION_CONTRACT:
        return model(texts, frames, **kwargs)
    if kwargs.get('crop_history_length') != 0:
        raise ValueError('Committed v3 requires adapter cache-only generation')
    with capture_committed_decode(model, frames, decoder_prefix) as outputs:
        model(texts, frames, **kwargs)
    return torch.cat(outputs, dim=1)
