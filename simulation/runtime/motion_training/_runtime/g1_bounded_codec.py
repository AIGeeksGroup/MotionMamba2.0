"""Explicit bounded codec contexts for isolated G1 contract diagnostics.

No installation at import and no changes to denoiser state or weights. These
helpers do not authorize changing the target representation of a checkpoint.
"""
from contextlib import contextmanager
import torch

TRAINING_CODEC_CONTRACT = 'g1_history_codec_104_v1'


def encode_training_history(model, motion, lengths, pad):
    """Versioned long teacher/generated history encoding; short tasks unchanged."""
    contract = getattr(model, 'training_codec_contract', 'legacy')
    if contract == 'legacy':
        return model.hybrid.get_hybrid_motion_from_explicit(motion, lengths, pad)
    if contract != TRAINING_CODEC_CONTRACT:
        raise ValueError(f'Unknown training codec contract: {contract}')
    if getattr(model, 'motion_contract', 'legacy') != 'g1_official_codec_v2':
        raise ValueError('Bounded history targets require corrected frozen codec statistics')
    return encode_windows(model.hybrid, motion, lengths, pad)


def encode_windows(converter, motion, lengths, pad, *, encode=None, horizon=52, history=52):
    encode = encode or converter.get_hybrid_motion_from_explicit
    patch = converter.num_frames_per_token
    if horizon <= 0 or history < 0 or horizon % patch or history % patch:
        raise ValueError('Codec contexts must be nonnegative patch-aligned frame counts')
    if motion.shape[1] % patch or pad.shape != motion.shape[:2]:
        raise ValueError('Expected patch-aligned explicit motion and matching padding')
    if not torch.equal(lengths, pad.sum(1)) or not pad.all():
        raise ValueError('Bounded diagnostic encoder currently requires all-real uniform histories')
    chunks=[];masks=[]
    for start in range(0,motion.shape[1],horizon):
        end=min(start+horizon,motion.shape[1])
        begin=max(0,start-history)
        window=pad[:,begin:end]
        tokens, tokenmask=encode(motion[:,begin:end],window.sum(1),window)
        take=(end-start)//patch
        chunks.append(tokens[:,-take:]);masks.append(tokenmask[:,-take:])
    return torch.cat(chunks,1),torch.cat(masks,1)


def decode_windows(converter, tokens, pad, lengths, motion_mask=None, *, decode=None, horizon=52, history=52):
    decode=decode or converter.get_explicit_motion_from_hybrid
    patch=converter.num_frames_per_token
    if horizon<=0 or history<0 or horizon%patch or history%patch:
        raise ValueError('Codec contexts must be patch-aligned')
    if tokens.shape[1]*patch != pad.shape[1] or not torch.equal(lengths,pad.sum(1)):
        raise ValueError('Decoder tokens, mask and lengths disagree')
    chunks=[]
    for start in range(0,pad.shape[1],horizon):
        end=min(start+horizon,pad.shape[1]);begin=max(0,start-history)
        window=pad[:,begin:end]
        result=decode(tokens[:,begin//patch:end//patch],window,window.sum(1),
                      motion_mask=None if motion_mask is None else motion_mask[:,begin:end])
        chunks.append(result[:,start-begin:end-begin])
    return torch.cat(chunks,1)


@contextmanager
def bounded_codec(model, *, encoder=False, decoder=True):
    """Temporarily bound explicit encoder/decoder calls in a diagnostic process."""
    converter=model.hybrid
    old_encode=converter.get_hybrid_motion_from_explicit
    old_decode=converter.get_explicit_motion_from_hybrid
    calls=[]
    def encode(motion,motion_len,motion_pad_mask):
        calls.append(dict(kind='encode',frames=motion.shape[1]))
        return encode_windows(converter,motion,motion_len,motion_pad_mask,encode=old_encode)
    def decode(hybrid_motion,motion_pad_mask,motion_len,motion_mask=None):
        calls.append(dict(kind='decode',frames=motion_pad_mask.shape[1]))
        return decode_windows(converter,hybrid_motion,motion_pad_mask,motion_len,motion_mask,decode=old_decode)
    if encoder: converter.get_hybrid_motion_from_explicit=encode
    if decoder: converter.get_explicit_motion_from_hybrid=decode
    try: yield calls
    finally:
        converter.get_hybrid_motion_from_explicit=old_encode
        converter.get_explicit_motion_from_hybrid=old_decode
