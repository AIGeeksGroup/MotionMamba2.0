# Copyright (c) 2025, Dao AI Lab, Goombalab
'Activation functions.'
import triton
import triton.language as tl

@triton.jit
def cos_approx(x):
    return tl.inline_asm_elementwise('cos.approx.f32 $0, $1;', constraints='=f,f', args=[x], dtype=tl.float32, is_pure=True, pack=1)

@triton.jit
def sin_approx(x):
    return tl.inline_asm_elementwise('sin.approx.f32 $0, $1;', constraints='=f,f', args=[x], dtype=tl.float32, is_pure=True, pack=1)

@triton.jit
def tanh_approx(x):
    return tl.inline_asm_elementwise('tanh.approx.f32 $0, $1;', constraints='=f,f', args=[x], dtype=tl.float32, is_pure=True, pack=1)

@triton.jit
def sech2_approx(x):
    tanh_x = tl.inline_asm_elementwise('tanh.approx.f32 $0, $1;', constraints='=f,f', args=[x], dtype=tl.float32, is_pure=True, pack=1)
    return 1.0 - tanh_x * tanh_x

@triton.jit
def sigmoid_approx(x):
    return tl.sigmoid(x)

@triton.jit
def silu(x):
    return x * tl.sigmoid(x)
