# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
'Backbone.'
import copy
import logging
from typing import Optional
from typing import Union
import torch
import torch.nn.functional as F
from omegaconf import ListConfig
from pydantic.dataclasses import dataclass
from torch import Tensor
from torch import nn
from motion_mamba2.model.tensor_utils import validate
log = logging.getLogger(__name__)

def pad_x_and_mask_to_fixed_size(x: Tensor, mask: Tensor, size: int):
    cur_max_size = x.shape[1]
    if cur_max_size == size:
        return (x, mask)
    if cur_max_size > size:
        log.warn('The size of the tensor is larger than the maximum size. Cropping the input..')
        return (x[:, :size], mask[:, :size])
    pad_len = size - cur_max_size
    new_x = torch.nn.functional.pad(x, (0, 0, 0, pad_len))
    new_mask = torch.nn.functional.pad(mask, (0, pad_len))
    return (new_x, new_mask)

def _get_activation_fn(activation):
    if callable(activation):
        return activation
    if activation == 'relu':
        return F.relu
    if activation == 'gelu':
        return F.gelu
    raise ValueError(f"Unsupported activation: {activation!r} (expected 'relu' or 'gelu')")

class SDPATransformerEncoderLayer(nn.Module):

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int=2048, dropout: float=0.1, activation='relu', layer_norm_eps: float=1e-05, batch_first: bool=True, norm_first: bool=False, bias: bool=True) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f'd_model ({d_model}) must be divisible by nhead ({nhead})')
        if not batch_first:
            raise NotImplementedError('Only batch_first=True is supported')
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.norm_first = norm_first
        self.attn_dropout_p = dropout
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=bias, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def _build_attn_mask(self, src, src_key_padding_mask, src_mask):
        if src_key_padding_mask is None and src_mask is None:
            return None
        (bs, seq_len, _) = src.shape
        attn_mask = torch.zeros(bs, 1, 1, seq_len, dtype=src.dtype, device=src.device)
        if src_key_padding_mask is not None:
            if src_key_padding_mask.dtype == torch.bool:
                attn_mask = attn_mask.masked_fill(src_key_padding_mask[:, None, None, :], float('-inf'))
            else:
                attn_mask = attn_mask + src_key_padding_mask[:, None, None, :]
        if src_mask is not None:
            if src_mask.dtype == torch.bool:
                add = torch.zeros_like(src_mask, dtype=src.dtype).masked_fill(src_mask, float('-inf'))
            else:
                add = src_mask.to(src.dtype)
            attn_mask = attn_mask + add
        return attn_mask

    def _sa_block(self, x, attn_mask):
        (bs, seq_len, _) = x.shape
        qkv = F.linear(x, self.self_attn.in_proj_weight, self.self_attn.in_proj_bias)
        (q, k, v) = qkv.chunk(3, dim=-1)
        q = q.view(bs, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(bs, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(bs, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        dropout_p = self.attn_dropout_p if self.training else 0.0
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p)
        attn = attn.transpose(1, 2).reshape(bs, seq_len, self.d_model)
        attn = self.self_attn.out_proj(attn)
        return self.dropout1(attn)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        if is_causal:
            raise NotImplementedError('is_causal is not supported')
        attn_mask = self._build_attn_mask(src, src_key_padding_mask, src_mask)
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), attn_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, attn_mask))
            x = self.norm2(x + self._ff_block(x))
        return x

class SDPATransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None, is_causal=None):
        out = src
        for layer in self.layers:
            out = layer(out, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            out = self.norm(out)
        return out

@dataclass(frozen=True, config=dict(extra='forbid', arbitrary_types_allowed=True))
class TransformerEncoderBlockConfig:
    input_dim: int
    output_dim: int
    skeleton: object
    llm_shape: Union[list[int], ListConfig]
    use_text_mask: bool
    latent_dim: int
    ff_size: int
    num_layers: int
    num_heads: int
    activation: str
    dropout: float
    pe_dropout: float
    norm_first: bool = False
    input_first_heading_angle: bool = False
    add_input_proj: bool = True
    positional_encoding_mode: str = 'default'

class TransformerEncoderBlock(nn.Module):

    @validate(TransformerEncoderBlockConfig, save_args=True, super_init=True)
    def __init__(self, conf):
        self.nbjoints = self.skeleton.nbjoints
        llm_dim = self.llm_shape[-1]
        self.embed_text = nn.Linear(llm_dim, self.latent_dim)
        self.num_text_tokens = self.llm_shape[0]
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.pe_dropout)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        if self.add_input_proj:
            self.input_linear = nn.Linear(self.input_dim, self.latent_dim)
        else:
            self.input_linear = nn.Identity()
        self.output_linear = nn.Linear(self.latent_dim, self.output_dim)
        if self.input_first_heading_angle:
            self.linear_first_heading_angle = nn.Linear(2, self.latent_dim)
        if self.positional_encoding_mode == 'learned_prefix_zero_at_first_generation':
            prefix_length = self.num_text_tokens + 1
            if self.input_first_heading_angle:
                prefix_length += 1
            self.prefix_length = prefix_length
            self.learned_prefix_embedding = LearnedPositionalEncoding(self.latent_dim, self.pe_dropout, max_len=prefix_length)
            self.motion_token_embedding = PositionalEncodingNegativeIndex(self.latent_dim, self.pe_dropout)
        elif self.positional_encoding_mode == 'default':
            pass
        else:
            raise ValueError(f'Invalid positional encoding mode: {self.positional_encoding_mode}')
        trans_enc_layer = SDPATransformerEncoderLayer(d_model=self.latent_dim, nhead=self.num_heads, dim_feedforward=self.ff_size, dropout=self.dropout, activation=self.activation, batch_first=True, norm_first=self.norm_first)
        self.seqTransEncoder = SDPATransformerEncoder(trans_enc_layer, num_layers=self.num_layers)

    def forward(self, x: Tensor, x_pad_mask: torch.Tensor, text_feat: torch.Tensor, text_feat_pad_mask: torch.Tensor, timesteps: Tensor, first_heading_angle: Optional[Tensor]=None, token_index: Optional[Tensor]=None) -> Tensor:
        batch_size = x.shape[0]
        x = self.input_linear(x)
        if self.num_text_tokens is not None:
            (text_feat, text_feat_pad_mask) = pad_x_and_mask_to_fixed_size(text_feat, text_feat_pad_mask, self.num_text_tokens)
        emb_text = self.embed_text(text_feat.to(self.embed_text.weight.dtype))
        emb_time = self.embed_timestep(timesteps)
        time_mask = torch.ones((batch_size, 1), dtype=bool, device=x.device)
        prefix_feats = torch.cat((emb_text, emb_time), axis=1)
        if not self.use_text_mask:
            text_feat_pad_mask = torch.ones((batch_size, emb_text.shape[1]), dtype=torch.bool, device=x.device)
        prefix_mask = torch.cat((text_feat_pad_mask, time_mask), axis=1)
        if self.input_first_heading_angle:
            assert first_heading_angle is not None, 'The first heading angle is mandatory for this model'
            first_heading_angle_feats = torch.stack([torch.cos(first_heading_angle), torch.sin(first_heading_angle)], axis=-1)
            first_heading_angle_feats = self.linear_first_heading_angle(first_heading_angle_feats)
            first_heading_angle_feats = first_heading_angle_feats[:, None]
            first_heading_angle_mask = torch.ones((batch_size, 1), dtype=bool, device=x.device)
            prefix_feats = torch.cat((prefix_feats, first_heading_angle_feats), axis=1)
            prefix_mask = torch.cat((prefix_mask, first_heading_angle_mask), axis=1)
        pose_start_ind = prefix_feats.shape[1]
        if self.positional_encoding_mode == 'default':
            xseq = torch.cat((prefix_feats, x), axis=1)
            xseq = self.sequence_pos_encoder(xseq)
        elif self.positional_encoding_mode == 'learned_prefix_zero_at_first_generation':
            prefix_feats_pe = self.learned_prefix_embedding(prefix_feats)
            x_pe = self.motion_token_embedding(x, token_index)
            xseq = torch.cat((prefix_feats_pe, x_pe), axis=1)
        src_key_padding_mask = ~torch.cat((prefix_mask, x_pad_mask), axis=1)
        output = self.seqTransEncoder(xseq, src_key_padding_mask=src_key_padding_mask)
        output = output[:, pose_start_ind:]
        output = self.output_linear(output)
        return output

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: Optional[float]=0.1, max_len: Optional[int]=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.pow(10000.0, -torch.arange(0, d_model, 2).float() / d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)

class TimestepEmbedder(nn.Module):

    def __init__(self, latent_dim: int, sequence_pos_encoder: PositionalEncoding):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder
        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(nn.Linear(self.latent_dim, time_embed_dim), nn.SiLU(), nn.Linear(time_embed_dim, time_embed_dim))

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.time_embed(F.embedding(timesteps.int(), self.sequence_pos_encoder.pe.squeeze(0))).unsqueeze(1)

class LearnedPositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout: Optional[float]=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding(max_len, d_model)
        self.max_len = max_len
        self.d_model = d_model

    def forward(self, x):
        assert x.shape[1] <= self.max_len, f'Input length {x.shape[1]} is greater than max length {self.max_len}'
        assert x.shape[2] == self.d_model, f'Input dimension {x.shape[2]} is not equal to d_model {self.d_model}'
        assert x.ndim == 3, f'Input dimension {x.ndim} is not 3'
        positions = torch.arange(0, x.shape[1], device=x.device, dtype=torch.int32).unsqueeze(0)
        x = x + self.embedding(positions)
        return self.dropout(x)

class PositionalEncodingNegativeIndex(nn.Module):

    def __init__(self, d_model: int, dropout: Optional[float]=0.1, max_len: Optional[int]=5000):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.pow(10000.0, -torch.arange(0, d_model, 2).float() / d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe_negative = torch.zeros(max_len - 1, d_model)
        pe_negative[:, 0::2] = torch.sin(-position[1:] * div_term)
        pe_negative[:, 1::2] = torch.cos(-position[1:] * div_term)
        pe_negative = torch.flip(pe_negative, dims=[0])
        pe = torch.cat([pe, pe_negative], dim=0)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        assert index.abs().max() < self.max_len, f'Index {index.abs().max()} is greater than max length {self.max_len}'
        safe_index = torch.where(index >= 0, index, index + self.pe.shape[0])
        positional_encoding = F.embedding(safe_index.int(), self.pe)
        x = x + positional_encoding
        return self.dropout(x)
