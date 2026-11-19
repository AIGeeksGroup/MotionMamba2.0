'Root body gradient.'
import math

def configure_root_body_gradient(raw, scale):
    scale = float(scale)
    if not math.isfinite(scale) or not 0 <= scale <= 1:
        raise ValueError('root_body_gradient_scale must be finite and in [0,1]')
    if scale and (not getattr(raw, 'supports_scaled_root_body_gradient', False)):
        raise ValueError('Loaded Motion Mamba 2.0 implementation does not support scaled root/body gradients')
    raw.root_body_gradient_scale = scale
