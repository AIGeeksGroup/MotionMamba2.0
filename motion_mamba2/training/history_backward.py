"""Order existing Mamba dK-state post-kernel calls; do not change its math.

The upstream post kernel atomically accumulates multiple value heads into a
shared GQA key head. History backpropagation needs d_ok_state and exposes this
otherwise unused reduction. Launching each batch/head on the same stream gives
the original atomic operations a fixed order, without editing Triton kernels.
"""
import importlib

_installed = False

def install():
    global _installed
    if _installed:
        return
    mod = importlib.import_module('motion_mamba2.kernels.mamba_backward')
    original = mod.apply_dk_state_post

    def ordered(d_ok_state, angles, k, k_bias, dk, dk_bias, dangles, Cu_Seqlens=None):
        if Cu_Seqlens is not None:
            raise ValueError('Unregistered varlen history backward; no silent fallback')
        batch, length, heads, _ = angles.shape
        key_heads = k.shape[2]
        assert heads % key_heads == 0
        ratio = heads // key_heads
        # Views retain original output storage; original wrapper reads their
        # strides, so no copies of output tensors or modified arithmetic.
        for b in range(batch):
            for h in range(heads):
                q = h // ratio
                original(
                    d_ok_state[b:b+1, h:h+1],
                    angles[b:b+1, :, h:h+1],
                    k[b:b+1, :, q:q+1], k_bias[h:h+1],
                    dk[b:b+1, :, q:q+1], dk_bias[h:h+1],
                    dangles[b:b+1, :, h:h+1], None,
                )
    mod.apply_dk_state_post = ordered
    _installed = True
    return {'ordered_history_backward': True,
            'original_triton_kernel_unchanged': True,
            'scope': 'non-varlen d_ok_state post accumulation only'}
