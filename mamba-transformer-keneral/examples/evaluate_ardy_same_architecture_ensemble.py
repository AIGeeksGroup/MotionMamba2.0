"""Stage-08 Minimal+Minimal versus Minimal+Mamba ensemble control."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "examples"))
from analyze_ardy_stage07_complementarity import _fit_scalar  # noqa: E402
from train_ardy_minimal_cached import (  # noqa: E402
    TrajectoryDataset, _commit_history, _move, _reset,
)


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--ardy-root", required=True); p.add_argument("--checkpoints-dir", required=True)
    p.add_argument("--minimal-a", required=True); p.add_argument("--minimal-b", required=True)
    p.add_argument("--complementarity-records", required=True)
    p.add_argument("--train-manifest", required=True); p.add_argument("--dev-manifest", required=True)
    p.add_argument("--output-dir", required=True); p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def _collect_b(raw, manifest: str, device: torch.device):
    data = TrajectoryDataset(manifest, None); records = []
    selected = {0, data.manifest["diffusion_steps"] // 2, data.manifest["diffusion_steps"] - 1}
    with torch.inference_mode():
        for ti, sample in enumerate(data.samples):
            steps = data.manifest["diffusion_steps"]
            for ci, call in enumerate(sample["calls"]):
                di = ci % steps
                if di not in selected: continue
                chunk = ci // steps; _reset(raw); _commit_history(raw, sample, chunk)
                for stage, block in (("root", raw.root_model), ("body", raw.body_model)):
                    inputs = _move(call[stage]["inputs"], device)
                    with torch.autocast("cuda", dtype=torch.bfloat16): prediction = block(*inputs)
                    records.append({"trajectory": ti, "chunk": chunk, "diffusion_index": di,
                                    "stage": stage, "prediction": prediction.float().cpu().flatten()})
            if ti == 0 or (ti + 1) % 8 == 0:
                print(json.dumps({"event":"minimal_b_progress","manifest":manifest,
                                  "trajectories_done":ti+1}), flush=True)
    _reset(raw); return records


def _join(parent, b):
    index = {(r["trajectory"],r["chunk"],r["diffusion_index"],r["stage"]):r for r in b}
    out=[]
    for r in parent:
        key=(r["trajectory"],r["chunk"],r["diffusion_index"],r["stage"])
        out.append({"minimal":r["minimal"],"mamba":index[key]["prediction"],"target":r["target"],
                    "stage":r["stage"]})
    return out


def _metrics(records, scalar):
    sums={k:0.0 for k in ("minimal_a","minimal_b","average","scalar")}
    for r in records:
        a,b,y=r["minimal"],r["mamba"],r["target"]
        pred={"minimal_a":a,"minimal_b":b,"average":.5*(a+b),
              "scalar":scalar*a+(1-scalar)*b}
        for k,v in pred.items(): sums[k]+=float(F.mse_loss(v.float(),y.float()))
    means={k:v/len(records) for k,v in sums.items()}; base=means["minimal_a"]
    return {"cases":len(records),"mean_mse":means,
            "relative_gain_vs_minimal_a":{k:(base-v)/base for k,v in means.items()}}


def main():
    args=_args(); device=torch.device(args.device); torch.cuda.set_device(device)
    sys.path.insert(0,str(Path(args.ardy_root).resolve()))
    from ardy.model import load_model
    from ardy_adapter import ArdySequenceKernelConfig, install_ardy_sequence_kernel
    model=load_model("Ardy-Core-RP-20FPS-Horizon40",device=args.device,text_encoder=False,
                     checkpoints_dir=args.checkpoints_dir)
    install_ardy_sequence_kernel(model,ArdySequenceKernelConfig(mode="minimal",attention_backend="sdpa",
                                    window_size_chunks=4,num_sink_tokens=4))
    raw=model.denoiser.model; ck=torch.load(args.minimal_b,map_location=device,weights_only=False)
    raw.root_model.load_state_dict(ck["root_model"],strict=True)
    raw.body_model.load_state_dict(ck["body_model"],strict=True); raw.eval()
    parent=torch.load(args.complementarity_records,map_location="cpu",weights_only=False)
    train=_join(parent["train"],_collect_b(raw,args.train_manifest,device))
    dev=_join(parent["development"],_collect_b(raw,args.dev_manifest,device))
    scalar=_fit_scalar(train)
    result={"phase":"stage08_same_architecture_control","final_heldout_used":False,
            "hashes":{"minimal_a":_sha(args.minimal_a),"minimal_b":_sha(args.minimal_b)},
            "scalar_weight_on_minimal_a":scalar,"train":_metrics(train,scalar),
            "development":_metrics(dev,scalar),
            "development_by_stage":{stage:_metrics([r for r in dev if r["stage"]==stage],scalar)
                                    for stage in ("root","body")}}
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    (out/"same_architecture.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
