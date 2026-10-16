"""ArdyMambaBlock: drop-in replacement for ARDY's `TransformerEncoderBlock`.

Milestone B of `docs/ardy_mamba_replacement_plan.md` -- "stateless denoiser
-block replacement." Keeps ARDY's windowing/diffusion loop, prefix
construction (text/timestep/heading-angle embedding), and positional
encoding EXACTLY as `ardy.model.backbone.TransformerEncoderBlock` does (by
importing and reusing its own embedding/positional-encoding classes, not
reimplementing them), and replaces ONLY the sequence mixer: the
bidirectional self-attention `SDPATransformerEncoder` stack becomes a stack
of causal Mamba-3 layers (via `ardy.model.mamba3_chunk_wrapper.Mamba3ChunkWrapper`),
each wrapped in a residual + LayerNorm the way `SDPATransformerEncoderLayer`
wraps its own self-attention and feed-forward sublayers -- with 8 layers
(the real "core" model's `denoiser.num_layers`), an unwrapped stack would
have no gradient-friendly skip path at all.

This is intentionally the FIRST, exploratory version, not a finished
integration -- see the caveats below and in
`docs/ardy_mamba_replacement_plan.md` Section 8.

Known simplifications (first version, not yet resolved):

1. **Causal, not bidirectional.** The original self-attention is
   non-causal within a window: every motion token attends to every other
   motion token (past AND future) plus the full prefix. Mamba's recurrence
   is inherently causal -- a motion token here only sees the prefix and
   EARLIER motion tokens, never later ones. This is a real representational
   difference, not a bug; it's exactly the tradeoff
   `ardy_mamba_replacement_plan.md` Section 8 asks to test empirically
   ("does it still generate coherent motion").
2. **Padding handled by zeroing, not masking.** Attention masks out padded
   positions from the attention weights directly; a linear recurrence has
   no equivalent operation. This block zeros a padded position's own
   CONTENT before the recurrence runs (`xseq * pad_mask`), so it
   contributes no information of its own, but (unlike real masking) a
   padded position still consumes a "processing step" and its zeroed
   content still passes through the recurrence's gating. Only benign if
   padding is a trailing suffix (padding_id positions after all real
   content) -- has NOT been checked against ARDY's actual padding
   convention (leading vs. trailing) at real batch scale.
3. **No shared weights with the original block.** This is a fresh module,
   not a checkpoint-compatible reparameterization -- it needs its own
   training, it cannot load `TransformerEncoderBlock`'s checkpoint weights
   for the mixer portion (the prefix/embedding/output-projection submodules
   below happen to have the same shapes, but are separate randomly
   -initialized parameters here).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _import_ardy_backbone():
    # Deferred import: this adapter depends on `ardy` being importable
    # (PYTHONPATH set to the ARDY repo).
    from ardy.model.backbone import (
        LearnedPositionalEncoding,
        PositionalEncoding,
        PositionalEncodingNegativeIndex,
        TimestepEmbedder,
        pad_x_and_mask_to_fixed_size,
    )

    return (
        LearnedPositionalEncoding,
        PositionalEncoding,
        PositionalEncodingNegativeIndex,
        TimestepEmbedder,
        pad_x_and_mask_to_fixed_size,
    )


class ArdyMambaBlock(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        skeleton=None,
        llm_shape,
        use_text_mask: bool,
        latent_dim: int,
        num_layers: int,
        pe_dropout: float = 0.0,
        input_first_heading_angle: bool = False,
        add_input_proj: bool = True,
        positional_encoding_mode: str = "default",
        mamba_headdim: int = 64,
        mamba_d_state: int = 64,
        mamba_expand: int = 2,
        mamba_chunk_size: int = 16,
        mamba_dtype: torch.dtype = torch.bfloat16,
        device=None,
        # Accepted for drop-in kwarg compatibility with
        # TransformerEncoderBlockConfig; not used by the Mamba mixer.
        ff_size=None,
        num_heads=None,
        activation=None,
        dropout: float = 0.0,
        norm_first: bool = False,
        **_ignored,
    ) -> None:
        super().__init__()
        (
            LearnedPositionalEncoding,
            PositionalEncoding,
            PositionalEncodingNegativeIndex,
            TimestepEmbedder,
            pad_x_and_mask_to_fixed_size,
        ) = _import_ardy_backbone()
        self._pad_x_and_mask_to_fixed_size = pad_x_and_mask_to_fixed_size

        self.nbjoints = getattr(skeleton, "nbjoints", None)
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

        if positional_encoding_mode == "learned_prefix_zero_at_first_generation":
            prefix_length = self.num_text_tokens + 1
            if input_first_heading_angle:
                prefix_length += 1
            self.prefix_length = prefix_length
            self.learned_prefix_embedding = LearnedPositionalEncoding(latent_dim, pe_dropout, max_len=prefix_length)
            self.motion_token_embedding = PositionalEncodingNegativeIndex(latent_dim, pe_dropout)
        elif positional_encoding_mode != "default":
            raise ValueError(f"Invalid positional encoding mode: {positional_encoding_mode!r}")

        # -- Mamba-3 sequence mixer stack, replacing SDPATransformerEncoder --
        from mamba_ssm.modules.mamba3 import Mamba3

        from ardy.model.mamba3_chunk_wrapper import Mamba3ChunkWrapper

        self.mamba_layers = nn.ModuleList(
            [
                Mamba3(
                    d_model=latent_dim,
                    d_state=mamba_d_state,
                    expand=mamba_expand,
                    headdim=mamba_headdim,
                    ngroups=1,
                    is_mimo=False,
                    chunk_size=mamba_chunk_size,
                    layer_idx=i,
                    device=device,
                    dtype=mamba_dtype,
                )
                for i in range(num_layers)
            ]
        )
        self._mamba_wrappers = [Mamba3ChunkWrapper(m) for m in self.mamba_layers]
        self.mamba_norms = nn.ModuleList([nn.LayerNorm(latent_dim) for _ in range(num_layers)])
        self._mamba_dtype = mamba_dtype

    def forward(
        self,
        x: Tensor,
        x_pad_mask: torch.Tensor,
        text_feat: torch.Tensor,
        text_feat_pad_mask: torch.Tensor,
        timesteps: Tensor,
        first_heading_angle: Optional[Tensor] = None,
        token_index: Optional[Tensor] = None,
    ) -> Tensor:
        """Identical signature/semantics to
        `ardy.model.backbone.TransformerEncoderBlock.forward` -- see that
        docstring for shapes. Returns `[B, T, output_dim]`."""
        batch_size = x.shape[0]
        x = self.input_linear(x)

        if self.num_text_tokens is not None:
            text_feat, text_feat_pad_mask = self._pad_x_and_mask_to_fixed_size(
                text_feat, text_feat_pad_mask, self.num_text_tokens
            )

        emb_text = self.embed_text(text_feat.to(self.embed_text.weight.dtype))
        emb_time = self.embed_timestep(timesteps)
        time_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=x.device)
        prefix_feats = torch.cat((emb_text, emb_time), dim=1)

        if not self.use_text_mask:
            text_feat_pad_mask = torch.ones((batch_size, emb_text.shape[1]), dtype=torch.bool, device=x.device)
        prefix_mask = torch.cat((text_feat_pad_mask, time_mask), dim=1)

        if self.input_first_heading_angle:
            assert first_heading_angle is not None, "first_heading_angle is mandatory for this model"
            first_heading_angle_feats = torch.stack(
                [torch.cos(first_heading_angle), torch.sin(first_heading_angle)], dim=-1
            )
            first_heading_angle_feats = self.linear_first_heading_angle(first_heading_angle_feats)[:, None]
            first_heading_angle_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=x.device)
            prefix_feats = torch.cat((prefix_feats, first_heading_angle_feats), dim=1)
            prefix_mask = torch.cat((prefix_mask, first_heading_angle_mask), dim=1)

        pose_start_ind = prefix_feats.shape[1]

        if self.positional_encoding_mode == "default":
            xseq = torch.cat((prefix_feats, x), dim=1)
            xseq = self.sequence_pos_encoder(xseq)
        else:  # "learned_prefix_zero_at_first_generation"
            prefix_feats_pe = self.learned_prefix_embedding(prefix_feats)
            x_pe = self.motion_token_embedding(x, token_index)
            xseq = torch.cat((prefix_feats_pe, x_pe), dim=1)

        # True = valid/attend, matching x_pad_mask's own convention -- see
        # module docstring point 2 for what this does and doesn't handle.
        full_valid_mask = torch.cat((prefix_mask, x_pad_mask), dim=1)
        xseq = xseq * full_valid_mask.unsqueeze(-1).to(xseq.dtype)

        h = xseq.to(self._mamba_dtype)
        for wrapper, norm in zip(self._mamba_wrappers, self.mamba_norms):
            mixed, _ = wrapper.forward_chunk(h, None)
            h = norm((h.float() + mixed.float())).to(self._mamba_dtype)

        output = h[:, pose_start_ind:].float()
        output = self.output_linear(output)
        return output
