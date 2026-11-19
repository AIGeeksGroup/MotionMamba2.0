"""Diffusion, decoded-body, visible-control and FK-consistency losses."""
import torch
FORWARD_KEYS = ["x", "history_len", "generation_len", "future_len", "history_mask", "generation_mask", "future_mask", "history_token_mask", "generation_token_mask", "future_token_mask", "text", "text_mask", "timestep", "heading", "motion_mask", "observed"]

def mean_valid(value,mask):
    mask=mask.expand_as(value).to(value);num=(value*mask).flatten(1).sum(1);den=mask.flatten(1).sum(1)
    return (num/den.clamp_min(1)).mean(),den
