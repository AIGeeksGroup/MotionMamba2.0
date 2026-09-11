"""Compare v5r/cache-fix checkpoints on identical non-sequential and sequential batches."""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import torch
from torch.utils.data import default_collate

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parents[0] / "ardy"), str(ROOT / "examples")]
for p in ("/data3/zts/packages/state-spaces-mamba", "/data3/zts/packages/mamba_op_triton"):
    sys.path.insert(0, p)
from ardy_data import HumanML3DArdyDataset
from train_humanml3d_ardy_tiny import _build_batch, _loss, _replace_stats, _load_denoiser_weights
from ardy.model import load_model
from ardy_adapter import ArdySequenceKernelConfig, install_ardy_sequence_kernel

BASE = Path("/data4/zts/result")
MAN = BASE / "humanml3d_ardy_train" / "manifest.jsonl"
DEV = BASE / "humanml3d_ardy_val" / "manifest.jsonl"
STATS = BASE / "humanml3d_ardy_train"
TEXT = Path("/data3/zts/datasets/HumanML3D/HumanML3D/texts")
CACHE = BASE / "humanml3d_text_cache_llm2vec"
RUNS = {
    "v5r_minimal": BASE / "humanml3d_ardy_full_v5r_minimal_20260819/minimal_humanml3d_ddp.pt",
    "cachefix_minimal": BASE / "humanml3d_full_cachefix_20260820_minimal/minimal_humanml3d_ddp.pt",
    "v5r_hybrid": BASE / "humanml3d_ardy_full_v5r_hybrid_20260819/hybrid_humanml3d_ddp.pt",
    "cachefix_hybrid": BASE / "humanml3d_full_cachefix_20260820_hybrid/hybrid_humanml3d_ddp.pt",
}

def make_fixed(mode: str, sequential: bool):
    ds = HumanML3DArdyDataset(str(DEV), str(STATS), text_root=str(TEXT), frames=80,
        horizon=40, condition_dropout=0.0, seed=1123, text_cache_dir=str(CACHE),
        history_probability=0.0)
    pool = ds.history_eligible_indices if sequential else list(range(len(ds)))
    items = [ds.sample(pool[i], 300000 + i, force_history=sequential) for i in range(4)]
    return default_collate(items)

@torch.no_grad()
def main():
    device = torch.device("cuda:0")
    out = {}
    batches = {flag: make_fixed("minimal", flag) for flag in (False, True)}
    for name, ckpt_path in RUNS.items():
        mode = name.split("_")[-1]
        model = load_model("core", device=str(device), text_encoder=False)
        _replace_stats(model, str(STATS))
        client = install_ardy_sequence_kernel(model, ArdySequenceKernelConfig(
            mode=mode, attention_backend="sdpa", window_size_chunks=4, num_sink_tokens=4))
        raw = model.denoiser.model
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        _load_denoiser_weights(raw, ckpt)
        raw.eval()
        out[name] = {}
        for sequential, batch in batches.items():
            fixed = _build_batch(model, batch, device, fixed_noise=True, sequential_cache=sequential)
            value = float(_loss(raw, client, fixed))
            out[name]["sequential" if sequential else "nonsequential"] = value
        del model
        torch.cuda.empty_cache()
    print(json.dumps(out, indent=2))
    Path("/data4/zts/result/cachefix_memory_diag_20260822/loss_cross_eval.json").write_text(json.dumps(out, indent=2) + "\n")

if __name__ == "__main__":
    main()
