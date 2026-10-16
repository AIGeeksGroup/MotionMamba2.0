"""Optional training gradient through the two-stage root condition.

This changes only backward propagation. Default zero retains the original
detached training condition; inference keeps its existing guidance gradients.
"""
import math
import torch


def validate_scale(scale):
    scale = float(scale)
    if not math.isfinite(scale) or not 0 <= scale <= 1:
        raise ValueError('root_body_gradient_scale must be finite and in [0,1]')
    return scale


class _ScaleGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, scale):
        ctx.scale = scale
        return value

    @staticmethod
    def backward(ctx, gradient):
        return gradient * ctx.scale, None


def training_root_condition(value, scale):
    scale = validate_scale(scale)
    if scale == 0:
        return value.detach()
    if scale == 1:
        return value
    return _ScaleGradient.apply(value, scale)
