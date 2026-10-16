"""Thin Stage-09 composition primitives over frozen Attention/Mamba parents."""

from __future__ import annotations

import torch
from torch import nn


class G0Correction(nn.Module):
    """The validated Stage-08 two-layer generic `[z, state]` correction."""

    def __init__(self, token_dim: int, state_dim: int, output_dim: int, width: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(token_dim + state_dim, width),
            nn.SiLU(),
            nn.Linear(width, output_dim),
        )

    def zero_output(self) -> None:
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, token: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((token, state), dim=-1))


class Stage09G0Block(nn.Module):
    """Run frozen stage parents on one input and apply the exact G0 family.

    This wrapper intentionally owns no new sequence/memory mechanism. Minimal
    KV lifecycle methods delegate to the Attention parent; the frozen Stage-07
    Mamba parent is the existing stateless denoiser replacement.
    """

    def __init__(
        self,
        attention: nn.Module,
        mamba: nn.Module,
        *,
        layer: int,
        state_source: str,
        output_dim: int,
        token_mean: torch.Tensor,
        token_std: torch.Tensor,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        width: int = 64,
        zero_output: bool = True,
    ) -> None:
        super().__init__()
        if state_source not in {"hidden", "recurrent"}:
            raise ValueError("state_source must be hidden or recurrent")
        self.attention = attention
        self.mamba = mamba
        self.layer = layer
        self.state_source = state_source
        self._attention_features: list[torch.Tensor] = []
        self._hook = attention.sequence_stack.layers[layer].register_forward_hook(
            self._capture_attention
        )
        mamba.enable_representation_capture(layers=(layer,))
        token_dim = int(token_mean.numel())
        state_dim = int(state_mean.numel())
        self.correction = G0Correction(token_dim, state_dim, output_dim, width)
        if zero_output:
            self.correction.zero_output()
        self.register_buffer("token_mean", token_mean.reshape(1, 1, -1).float())
        self.register_buffer("token_std", token_std.reshape(1, 1, -1).float().clamp_min(1e-4))
        self.register_buffer("state_mean", state_mean.reshape(1, 1, -1).float())
        self.register_buffer("state_std", state_std.reshape(1, 1, -1).float().clamp_min(1e-4))
        self.last_output: torch.Tensor | None = None
        self.last_token: torch.Tensor | None = None
        self.last_state: torch.Tensor | None = None
        self.last_attention_output: torch.Tensor | None = None
        self.last_valid: torch.Tensor | None = None
        self.attention_forwards = 0
        self.mamba_forwards = 0
        for parent in (attention, mamba):
            parent.eval()
            for parameter in parent.parameters():
                parameter.requires_grad_(False)

    def _capture_attention(self, _module, _inputs, output) -> None:
        self._attention_features.append(output[0].detach())

    def _state_feature(self, token_count: int) -> torch.Tensor:
        captured = self.mamba.captured_representations
        if self.state_source == "hidden":
            return captured["token_hidden"][self.layer].float()
        state = captured["ssm_state_summary"][self.layer].flatten(1).float()
        position = torch.linspace(-1, 1, token_count, device=state.device, dtype=state.dtype)
        position = position.reshape(1, token_count, 1).expand(state.shape[0], -1, -1)
        return torch.cat((state[:, None, :].expand(-1, token_count, -1), position), dim=-1)

    def forward(self, *inputs):
        self._attention_features.clear()
        with torch.no_grad():
            attention_output = self.attention(*inputs)
            self.attention_forwards += 1
            _ = self.mamba(*inputs)
            self.mamba_forwards += 1
        # Accepted-history commits trigger an additional short internal stack
        # call. Keep only read-path features matching the parent block output;
        # separated-CFG read calls are then concatenated back over batch.
        read_features = [
            feature for feature in self._attention_features
            if feature.shape[1] == attention_output.shape[1]
        ]
        token = torch.cat(read_features, dim=0).float()
        state = self._state_feature(token.shape[1])
        if token.shape[:2] != state.shape[:2] or token.shape[:2] != attention_output.shape[:2]:
            raise RuntimeError(
                f"G0 feature alignment failed: token={tuple(token.shape)}, "
                f"state={tuple(state.shape)}, output={tuple(attention_output.shape)}"
            )
        token = (token - self.token_mean) / self.token_std
        state = (state - self.state_mean) / self.state_std
        correction = self.correction(token, state)
        valid = inputs[1].to(correction.device).unsqueeze(-1)
        output = attention_output.float() + correction * valid
        self.last_token = token.detach()
        self.last_state = state.detach()
        self.last_attention_output = attention_output.detach().float()
        self.last_valid = inputs[1].detach().bool()
        self.last_output = output
        return output

    def reset_memory(self, reset_mask=None) -> None:
        self.attention.reset_memory(reset_mask)

    def prepare_accepted_commit(self, selection, *, chunk_id, detach=True) -> None:
        self.attention.prepare_accepted_commit(selection, chunk_id=chunk_id, detach=detach)

    def _stack_parameter(self):
        return self.attention._stack_parameter()

    @property
    def sequence_stack(self):
        """Expose the delegated Attention cache configuration to lifecycle helpers."""
        return self.attention.sequence_stack

    @property
    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.correction.parameters())

    def correction_state_dict(self):
        return self.correction.state_dict()

    def load_correction_state_dict(self, state_dict) -> None:
        self.correction.load_state_dict(state_dict, strict=True)
