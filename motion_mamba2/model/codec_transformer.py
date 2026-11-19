# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
'Codec transformer.'
from typing import Optional
import torch
import torch.nn as nn
from einops import rearrange

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

class EncoderTransformer(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, num_frames_per_token: int, latent_dim: int, num_heads: int, ff_size: int, dropout: float, activation: str, norm_first: bool, num_layers: int, pe_dropout: float, is_causal: bool=True):
        super().__init__()
        self.is_causal = is_causal
        self.num_frames_per_token = num_frames_per_token
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.ff_size = ff_size
        self.dropout = dropout
        self.activation = activation
        self.norm_first = norm_first
        self.num_layers = num_layers
        self.pe_dropout = pe_dropout
        self.input_proj = nn.Linear(input_dim * num_frames_per_token, latent_dim)
        self.sequence_pos_encoder = PositionalEncoding(latent_dim, pe_dropout)
        trans_enc_layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=num_heads, dim_feedforward=ff_size, dropout=dropout, activation=activation, batch_first=True, norm_first=norm_first)
        self.seqTransEncoder = nn.TransformerEncoder(trans_enc_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.output_proj = nn.Linear(latent_dim, output_dim)

    def forward(self, x, motion_pad_mask: Optional[torch.BoolTensor]=None):
        x = rearrange(x, 'b (t f) d -> b t (f d)', f=self.num_frames_per_token)
        x = self.input_proj(x)
        x = self.sequence_pos_encoder(x)
        attention_mask = torch.nn.Transformer.generate_square_subsequent_mask(x.shape[1], device=x.device, dtype=torch.bool) if self.is_causal else None
        if motion_pad_mask is not None:
            motion_pad_mask = rearrange(motion_pad_mask, 'b (t f) -> b t f', f=self.num_frames_per_token)
            token_pad_mask = (motion_pad_mask > 0.5).all(dim=-1)
            src_key_padding_mask = ~token_pad_mask
        else:
            src_key_padding_mask = None
        x = self.seqTransEncoder(x, mask=attention_mask, src_key_padding_mask=src_key_padding_mask)
        output_embeddings = self.output_proj(x)
        return output_embeddings

class DoubleCondDecoderTransformer(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, num_frames_per_token: int, latent_dim: int, num_heads: int, ff_size: int, dropout: float, activation: str, norm_first: bool, num_layers: int, pe_dropout: float, is_causal: bool=True, target_cond_dim: int=-1, external_cond_dim: int=-1):
        super().__init__()
        self.is_causal = is_causal
        self.num_frames_per_token = num_frames_per_token
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.ff_size = ff_size
        self.dropout = dropout
        self.activation = activation
        self.norm_first = norm_first
        self.num_layers = num_layers
        self.pe_dropout = pe_dropout
        (self._target_cond_dim, self._external_cond_dim) = (target_cond_dim, external_cond_dim)
        (self._HAS_EXTERNAL_COND, self._HAS_TARGET_COND) = (external_cond_dim > 0, target_cond_dim > 0)
        self.input_proj = nn.Linear(input_dim, latent_dim)
        self.sequence_pos_encoder = PositionalEncoding(latent_dim, pe_dropout)
        trans_enc_layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=num_heads, dim_feedforward=ff_size, dropout=dropout, activation=activation, batch_first=True, norm_first=norm_first)
        self.seqTransEncoder = nn.TransformerEncoder(trans_enc_layer, num_layers=num_layers, enable_nested_tensor=False)
        'Step 2: external cond embeddings.\n\n        At each layer, the external embedding will be merged with the hidden state. The external\n        condition is dense and expected to be always available in each frame.\n        '
        if self._HAS_EXTERNAL_COND:
            external_cond_blocks = []
            cond_feat_dim = num_frames_per_token * self._external_cond_dim
            external_cond_blocks.append(nn.Linear(cond_feat_dim + latent_dim, latent_dim))
            external_cond_blocks.append(nn.ReLU())
            self.external_cond_blocks = nn.Sequential(*external_cond_blocks)
        'Step 3: target cond embeddings.\n\n        The target condition is sparse and expected to be available in certain frames. we replace\n        the hidden state with the target condition embeddings for given frames. In earlier layers\n        where each position corresponds to multiple frames, we reshape the hidden states to map to\n        each frame position (see @forward method)\n        '
        if self._HAS_TARGET_COND:
            target_cond_blocks = []
            assert latent_dim % num_frames_per_token == 0, 'latent_dim % num_frames_per_token needs to 0 so that hidden can be split for each frame in earlier layers.'
            target_cond_blocks.append(nn.Linear(self._target_cond_dim, int(latent_dim / num_frames_per_token)))
            target_cond_blocks.append(nn.ReLU())
            self.target_cond_blocks = nn.Sequential(*target_cond_blocks)
        self.output_proj = nn.Linear(latent_dim, output_dim * num_frames_per_token)

    def forward(self, x: torch.Tensor, external_cond: torch.Tensor=None, target_cond: Optional[torch.Tensor]=None, has_target_cond: Optional[torch.Tensor]=None, motion_pad_mask: Optional[torch.BoolTensor]=None):
        batch_size = x.shape[0]
        num_tokens = x.shape[1]
        num_frames_per_token = self.num_frames_per_token
        num_frames = num_tokens * num_frames_per_token
        h = self.input_proj(x)
        if not self._HAS_TARGET_COND or target_cond is None or has_target_cond is None:
            pass
        else:
            h_target_cond = self.target_cond_blocks(target_cond)
            h = h.reshape([batch_size, num_frames, self.latent_dim // self.num_frames_per_token])
            h = torch.where(has_target_cond[:, :, None], h_target_cond, h)
            h = h.reshape([batch_size, num_tokens, self.latent_dim])
        if self._HAS_EXTERNAL_COND:
            assert external_cond is not None and external_cond.shape[1] == num_frames
            external_cond = external_cond.reshape([batch_size, num_tokens, -1])
            h = torch.cat([h, external_cond], dim=-1)
            h = self.external_cond_blocks(h)
        h = self.sequence_pos_encoder(h)
        attention_mask = torch.nn.Transformer.generate_square_subsequent_mask(h.shape[1], device=h.device, dtype=torch.bool) if self.is_causal else None
        if motion_pad_mask is not None:
            motion_pad_mask = rearrange(motion_pad_mask, 'b (t f) -> b t f', f=self.num_frames_per_token)
            token_pad_mask = (motion_pad_mask > 0.5).all(dim=-1)
            src_key_padding_mask = ~token_pad_mask
        else:
            src_key_padding_mask = None
        h = self.seqTransEncoder(h, mask=attention_mask, src_key_padding_mask=src_key_padding_mask)
        h = self.output_proj(h)
        h = rearrange(h, 'b t (f d) -> b (t f) d', f=self.num_frames_per_token)
        return h
