"""Load saved Minimal/Transformer weights without historical experiment imports."""
import random
import inspect
import numpy as np
import torch
from paths import STATS


class StatelessClient:
    def reset_memory(self):
        pass


def load(checkpoint, kind, device="cuda:0"):
    from motion_training import runtime
    from motion_training._compat import legacy_loss
    from g1_motion_contract import inspect_stats, validate_checkpoint
    from g1_padding_contract import install_padding_contract
    from g1_root_body_gradient import configure_root_body_gradient
    from g1_training_precision import configure_training_precision, FP32_BOUNDARIES
    from ardy_adapter import ArdySequenceKernelConfig, install_ardy_sequence_kernel
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    ck = dict(ck, config=dict(ck["config"]))
    cfg = ck["config"]
    cfg["stats_dir"] = str(STATS)
    validate_checkpoint(ck, inspect_stats(STATS))
    random.seed(123); np.random.seed(123); torch.manual_seed(123); torch.cuda.manual_seed_all(123)
    kwargs = {"fresh_denoiser": True} if "fresh_denoiser" in inspect.signature(runtime.load_model).parameters else {}
    model = runtime.load_model("g1", device=device, text_encoder=False, **kwargs)
    client = StatelessClient()
    if kind == "minimal":
        client = install_ardy_sequence_kernel(model, ArdySequenceKernelConfig(
            mode="minimal", attention_backend="sdpa", window_size_chunks=4,
            num_sink_tokens=4, content_memory_enabled=False))
    raw = model.denoiser.model
    legacy_loss._load_denoiser_weights(raw, ck)
    legacy_loss._replace_stats(model, str(STATS))
    for key in ("training_codec_contract", "generation_contract"):
        setattr(model, key, cfg.get(key, "legacy"))
    model.checkpoint_training_manifest = cfg.get("manifest")
    install_padding_contract(model, cfg.get("padding_contract", "legacy"))
    configure_root_body_gradient(raw, cfg.get("root_body_gradient_scale", 0))
    configure_training_precision(raw, model.autoencoder, FP32_BOUNDARIES)
    model.device = next(raw.parameters()).device
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model, client, ck
