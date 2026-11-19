# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
'Guidance.'
from typing import Optional
import torch
import torch.nn as nn
CFG_TYPES = ['nocfg', 'regular', 'separated']

def AutoLatentClassifierFreeGuidedModel(model: nn.Module, cfg_type: Optional[str]='separated'):
    if cfg_type == 'nocfg':
        return AutoLatentClassifierFreeGuidedModelNoCFG(model)
    elif cfg_type == 'regular':
        return AutoLatentClassifierFreeGuidedModelRegular(model)
    elif cfg_type == 'separated':
        return AutoLatentClassifierFreeGuidedModelSeparated(model)
    raise ValueError(f'Unknown cfg_type {cfg_type!r}. Available: {CFG_TYPES}')

class AutoLatentClassifierFreeGuidedModelSeparated(nn.Module):

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def forward(self, cfg_weight_text: torch.Tensor, cfg_weight_cstr: torch.Tensor, x: torch.Tensor, history_len: torch.Tensor, generation_len: torch.Tensor, future_len: torch.Tensor, history_mask: torch.Tensor, generation_mask: torch.Tensor, future_mask: torch.Tensor, history_token_mask: torch.Tensor, generation_token_mask: torch.Tensor, future_token_mask: torch.Tensor, text_feat: torch.Tensor, text_feat_pad_mask: torch.Tensor, timesteps: torch.Tensor, first_heading_angle: torch.Tensor, motion_mask: torch.Tensor, observed_motion: torch.Tensor) -> torch.Tensor:
        x_3 = torch.cat([x, x, x], dim=0)
        history_len_3 = torch.cat([history_len, history_len, history_len], dim=0)
        generation_len_3 = torch.cat([generation_len, generation_len, generation_len], dim=0)
        future_len_3 = torch.cat([future_len, future_len, future_len], dim=0)
        history_mask_3 = torch.cat([history_mask, history_mask, history_mask], dim=0)
        generation_mask_3 = torch.cat([generation_mask, generation_mask, generation_mask], dim=0)
        future_mask_3 = torch.cat([future_mask, future_mask, future_mask], dim=0)
        history_token_mask_3 = torch.cat([history_token_mask, history_token_mask, history_token_mask], dim=0)
        generation_token_mask_3 = torch.cat([generation_token_mask, generation_token_mask, generation_token_mask], dim=0)
        future_token_mask_3 = torch.cat([0 * future_token_mask, future_token_mask, 0 * future_token_mask], dim=0)
        text_feat_3 = torch.cat([text_feat, 0 * text_feat, 0 * text_feat], dim=0)
        text_feat_pad_mask_3 = torch.cat([text_feat_pad_mask, 0 * text_feat_pad_mask, 0 * text_feat_pad_mask], dim=0)
        timesteps_3 = torch.cat([timesteps, timesteps, timesteps], dim=0)
        first_heading_angle_3 = torch.cat([first_heading_angle, first_heading_angle, first_heading_angle], dim=0) if first_heading_angle is not None else None
        motion_mask_3 = torch.cat([0 * motion_mask, motion_mask, 0 * motion_mask], dim=0) if motion_mask is not None else None
        observed_motion_3 = torch.cat([0 * observed_motion, observed_motion, 0 * observed_motion], dim=0) if observed_motion is not None else None
        out_3 = self.model(x=x_3, history_len=history_len_3, generation_len=generation_len_3, future_len=future_len_3, history_mask=history_mask_3 > 0.5, generation_mask=generation_mask_3 > 0.5, future_mask=future_mask_3 > 0.5, history_token_mask=history_token_mask_3 > 0.5, generation_token_mask=generation_token_mask_3 > 0.5, future_token_mask=future_token_mask_3 > 0.5, text_feat=text_feat_3, text_feat_pad_mask=text_feat_pad_mask_3 > 0.5, timesteps=timesteps_3, first_heading_angle=first_heading_angle_3, motion_mask=motion_mask_3, observed_motion=observed_motion_3)
        (out_text, out_cstr, out_uncond) = torch.chunk(out_3, 3, dim=0)
        return out_uncond + cfg_weight_text * (out_text - out_uncond) + cfg_weight_cstr * (out_cstr - out_uncond)

class AutoLatentClassifierFreeGuidedModelRegular(nn.Module):

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def forward(self, cfg_weight_text: torch.Tensor, cfg_weight_cstr: torch.Tensor, x: torch.Tensor, history_len: torch.Tensor, generation_len: torch.Tensor, future_len: torch.Tensor, history_mask: torch.Tensor, generation_mask: torch.Tensor, future_mask: torch.Tensor, history_token_mask: torch.Tensor, generation_token_mask: torch.Tensor, future_token_mask: torch.Tensor, text_feat: torch.Tensor, text_feat_pad_mask: torch.Tensor, timesteps: torch.Tensor, first_heading_angle: torch.Tensor=None, motion_mask: torch.Tensor=None, observed_motion: torch.Tensor=None) -> torch.Tensor:
        text_feat = torch.cat([text_feat, 0 * text_feat], dim=0)
        if motion_mask is not None:
            motion_mask = torch.cat([motion_mask, 0 * motion_mask], dim=0)
        if observed_motion is not None:
            observed_motion = torch.cat([observed_motion, 0 * observed_motion], dim=0)
        if first_heading_angle is not None:
            first_heading_angle = torch.cat([first_heading_angle, first_heading_angle], dim=0)
        out_cond_uncond = self.model(torch.cat([x, x], dim=0), torch.cat([history_len, history_len], dim=0), torch.cat([generation_len, generation_len], dim=0), torch.cat([future_len, future_len], dim=0), torch.cat([history_mask, history_mask], dim=0), torch.cat([generation_mask, generation_mask], dim=0), torch.cat([future_mask, future_mask], dim=0), torch.cat([history_token_mask, history_token_mask], dim=0), torch.cat([generation_token_mask, generation_token_mask], dim=0), torch.cat([future_token_mask, False * future_token_mask], dim=0) if future_token_mask is not None else None, text_feat, torch.cat([text_feat_pad_mask, False * text_feat_pad_mask], dim=0), torch.cat([timesteps, timesteps], dim=0), first_heading_angle=first_heading_angle, motion_mask=motion_mask, observed_motion=observed_motion)
        (out, out_uncond) = torch.chunk(out_cond_uncond, 2)
        return out_uncond + cfg_weight_text * (out - out_uncond)

class AutoLatentClassifierFreeGuidedModelNoCFG(nn.Module):

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def forward(self, cfg_weight_text: torch.Tensor, cfg_weight_cstr: torch.Tensor, x: torch.Tensor, history_len: torch.Tensor, generation_len: torch.Tensor, future_len: torch.Tensor, history_mask: torch.Tensor, generation_mask: torch.Tensor, future_mask: torch.Tensor, history_token_mask: torch.Tensor, generation_token_mask: torch.Tensor, future_token_mask: torch.Tensor, text_feat: torch.Tensor, text_feat_pad_mask: torch.Tensor, timesteps: torch.Tensor, first_heading_angle: torch.Tensor=None, motion_mask: torch.Tensor=None, observed_motion: torch.Tensor=None) -> torch.Tensor:
        if motion_mask is not None:
            motion_mask = motion_mask.to(dtype=x.dtype)
        return self.model(x, history_len, generation_len, future_len, history_mask, generation_mask, future_mask, history_token_mask, generation_token_mask, future_token_mask, text_feat, text_feat_pad_mask, timesteps, first_heading_angle=first_heading_angle, motion_mask=motion_mask, observed_motion=observed_motion)
