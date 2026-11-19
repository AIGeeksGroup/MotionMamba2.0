'Mamba block.'
from __future__ import annotations
from typing import Optional
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

def _import_ardy_backbone():
    from motion_mamba2.model.backbone import LearnedPositionalEncoding
    from motion_mamba2.model.backbone import PositionalEncoding
    from motion_mamba2.model.backbone import PositionalEncodingNegativeIndex
    from motion_mamba2.model.backbone import TimestepEmbedder
    from motion_mamba2.model.backbone import pad_x_and_mask_to_fixed_size
    return (LearnedPositionalEncoding, PositionalEncoding, PositionalEncodingNegativeIndex, TimestepEmbedder, pad_x_and_mask_to_fixed_size)

class MotionMambaBlock(nn.Module):

    def __init__(self, *, input_dim: int, output_dim: int, skeleton=None, llm_shape, use_text_mask: bool, latent_dim: int, num_layers: int, pe_dropout: float=0.0, input_first_heading_angle: bool=False, add_input_proj: bool=True, positional_encoding_mode: str='default', mamba_headdim: int=64, mamba_d_state: int=64, mamba_expand: int=2, mamba_chunk_size: int=16, mamba_dtype: torch.dtype=torch.bfloat16, device=None, ff_size=None, num_heads=None, activation=None, dropout: float=0.0, norm_first: bool=False, **_ignored) -> None:
        super().__init__()
        (LearnedPositionalEncoding, PositionalEncoding, PositionalEncodingNegativeIndex, TimestepEmbedder, pad_x_and_mask_to_fixed_size) = _import_ardy_backbone()
        self._pad_x_and_mask_to_fixed_size = pad_x_and_mask_to_fixed_size
        self.nbjoints = getattr(skeleton, 'nbjoints', None)
        llm_dim = llm_shape[-1]
        self.num_text_tokens = llm_shape[0]
        self.latent_dim = latent_dim
        self.use_text_mask = use_text_mask
        self.input_first_heading_angle = input_first_heading_angle
        self.positional_encoding_mode = positional_encoding_mode
        self.embed_text = nn.Linear(llm_dim, latent_dim)
        self.sequence_pos_encoder = PositionalEncoding(latent_dim, pe_dropout)
        self.embed_timestep = TimestepEmbedder(latent_dim, self.sequence_pos_encoder)
        self.input_linear = nn.Linear(input_dim, latent_dim) if add_input_proj else nn.Identity()
        self.output_linear = nn.Linear(latent_dim, output_dim)
        if input_first_heading_angle:
            self.linear_first_heading_angle = nn.Linear(2, latent_dim)
        if positional_encoding_mode == 'learned_prefix_zero_at_first_generation':
            prefix_length = self.num_text_tokens + 1
            if input_first_heading_angle:
                prefix_length += 1
            self.prefix_length = prefix_length
            self.learned_prefix_embedding = LearnedPositionalEncoding(latent_dim, pe_dropout, max_len=prefix_length)
            self.motion_token_embedding = PositionalEncodingNegativeIndex(latent_dim, pe_dropout)
        elif positional_encoding_mode != 'default':
            raise ValueError(f'Invalid positional encoding mode: {positional_encoding_mode!r}')
        from motion_mamba2.model.mamba import Mamba3
        from motion_mamba2.model.mamba_state import Mamba3ChunkWrapper
        self.mamba_layers = nn.ModuleList([Mamba3(d_model=latent_dim, d_state=mamba_d_state, expand=mamba_expand, headdim=mamba_headdim, ngroups=1, is_mimo=False, chunk_size=mamba_chunk_size, layer_idx=i, device=device, dtype=mamba_dtype) for i in range(num_layers)])
        self._mamba_wrappers = [Mamba3ChunkWrapper(m) for m in self.mamba_layers]
        self.mamba_norms = nn.ModuleList([nn.LayerNorm(latent_dim) for _ in range(num_layers)])
        self._mamba_dtype = mamba_dtype
        self._capture_representations = False
        self._capture_layers: set[int] = set()
        self.captured_representations: dict[str, dict[int, Tensor]] = {}

    def enable_representation_capture(self, enabled: bool=True, layers: Optional[tuple[int, ...]]=None) -> None:
        self._capture_representations = enabled
        self._capture_layers = set(range(len(self.mamba_layers))) if layers is None else set(layers)
        if any((index < 0 or index >= len(self.mamba_layers) for index in self._capture_layers)):
            raise ValueError('capture layer index out of range')
        self.captured_representations = {}

    def forward(self, x: Tensor, x_pad_mask: torch.Tensor, text_feat: torch.Tensor, text_feat_pad_mask: torch.Tensor, timesteps: Tensor, first_heading_angle: Optional[Tensor]=None, token_index: Optional[Tensor]=None) -> Tensor:
        batch_size = x.shape[0]
        x = self.input_linear(x)
        if self.num_text_tokens is not None:
            (text_feat, text_feat_pad_mask) = self._pad_x_and_mask_to_fixed_size(text_feat, text_feat_pad_mask, self.num_text_tokens)
        emb_text = self.embed_text(text_feat.to(self.embed_text.weight.dtype))
        emb_time = self.embed_timestep(timesteps)
        time_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=x.device)
        prefix_feats = torch.cat((emb_text, emb_time), dim=1)
        if not self.use_text_mask:
            text_feat_pad_mask = torch.ones((batch_size, emb_text.shape[1]), dtype=torch.bool, device=x.device)
        prefix_mask = torch.cat((text_feat_pad_mask, time_mask), dim=1)
        if self.input_first_heading_angle:
            assert first_heading_angle is not None, 'first_heading_angle is mandatory for this model'
            first_heading_angle_feats = torch.stack([torch.cos(first_heading_angle), torch.sin(first_heading_angle)], dim=-1)
            first_heading_angle_feats = self.linear_first_heading_angle(first_heading_angle_feats)[:, None]
            first_heading_angle_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=x.device)
            prefix_feats = torch.cat((prefix_feats, first_heading_angle_feats), dim=1)
            prefix_mask = torch.cat((prefix_mask, first_heading_angle_mask), dim=1)
        pose_start_ind = prefix_feats.shape[1]
        if self.positional_encoding_mode == 'default':
            xseq = torch.cat((prefix_feats, x), dim=1)
            xseq = self.sequence_pos_encoder(xseq)
        else:
            prefix_feats_pe = self.learned_prefix_embedding(prefix_feats)
            x_pe = self.motion_token_embedding(x, token_index)
            xseq = torch.cat((prefix_feats_pe, x_pe), dim=1)
        full_valid_mask = torch.cat((prefix_mask, x_pad_mask), dim=1)
        xseq = xseq * full_valid_mask.unsqueeze(-1).to(xseq.dtype)
        h = xseq.to(self._mamba_dtype)
        captured_hidden = {}
        captured_state = {}
        for (layer_index, (wrapper, norm)) in enumerate(zip(self._mamba_wrappers, self.mamba_norms)):
            (mixed, state) = wrapper.forward_chunk(h, None)
            h = norm(h.float() + mixed.float()).to(self._mamba_dtype)
            if self._capture_representations and layer_index in self._capture_layers:
                captured_hidden[layer_index] = h[:, pose_start_ind:].detach()
                captured_state[layer_index] = state.ssm_state.detach().float().mean(dim=-1)
        if self._capture_representations:
            self.captured_representations = {'token_hidden': captured_hidden, 'ssm_state_summary': captured_state}
        output = h[:, pose_start_ind:].float()
        output = self.output_linear(output)
        return output
