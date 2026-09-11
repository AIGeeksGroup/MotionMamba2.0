"""Extract frozen middle/final Minimal and Mamba representations for Stage 08."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "examples")]
from train_ardy_mamba_cached import _make_blocks  # noqa: E402
from train_ardy_minimal_cached import (  # noqa: E402
    TrajectoryDataset, _commit_history, _move, _reset,
)


def _args():
    p=argparse.ArgumentParser(); p.add_argument("--ardy-root",required=True)
    p.add_argument("--checkpoints-dir",required=True); p.add_argument("--minimal-checkpoint",required=True)
    p.add_argument("--mamba-checkpoint",required=True); p.add_argument("--train-manifest",required=True)
    p.add_argument("--dev-manifest",required=True); p.add_argument("--output-dir",required=True)
    p.add_argument("--device",default="cuda:0"); p.add_argument("--layers",type=int,nargs="+",default=[3,7])
    return p.parse_args()


class _LayerCapture:
    def __init__(self, block, layers):
        self.values={}; self.handles=[]
        for index in layers:
            self.handles.append(block.sequence_stack.layers[index].register_forward_hook(
                lambda _module,_inputs,output,index=index:self.values.setdefault(index,[]).append(output[0].detach())
            ))
    def clear(self):
        self.values.clear()

    def get(self,index):
        return torch.cat(self.values[index],dim=0)
    def close(self):
        for handle in self.handles: handle.remove()


def _collect(raw,mroot,mbody,manifest,device,layers):
    data=TrajectoryDataset(manifest,None); buckets={stage:{layer:default_bucket() for layer in layers}
                                                    for stage in ("root","body")}
    captures={"root":_LayerCapture(raw.root_model,layers),"body":_LayerCapture(raw.body_model,layers)}
    next_sequence={"root":0,"body":0}
    mroot.enable_representation_capture(layers=tuple(layers)); mbody.enable_representation_capture(layers=tuple(layers))
    selected={0,data.manifest["diffusion_steps"]//2,data.manifest["diffusion_steps"]-1}
    with torch.inference_mode():
        for trajectory,sample in enumerate(data.samples):
            steps=data.manifest["diffusion_steps"]
            for call_index,call in enumerate(sample["calls"]):
                diffusion=call_index%steps
                if diffusion not in selected: continue
                chunk=call_index//steps; _reset(raw); _commit_history(raw,sample,chunk)
                for stage,ablock,mblock in (("root",raw.root_model,mroot),("body",raw.body_model,mbody)):
                    inputs=_move(call[stage]["inputs"],device); valid=inputs[1].bool()
                    # The Minimal adapter evaluates separated-CFG B rows in
                    # several state-sized calls.  Accumulate every hook result
                    # and concatenate them back to the original input order.
                    captures[stage].clear()
                    with torch.autocast("cuda",dtype=torch.bfloat16):
                        amin=ablock(*inputs); _=mblock(*inputs)
                    residual=call[stage]["teacher_output"].to(device).float()-amin.float()
                    sequence=(torch.arange(z_batch:=valid.shape[0],device=device)+next_sequence[stage])[:,None].expand(-1,valid.shape[1])
                    token_position=torch.arange(valid.shape[1],device=device)[None,:].expand(z_batch,-1)
                    next_sequence[stage]+=z_batch
                    for layer in layers:
                        z=captures[stage].get(layer).float()
                        mh=mblock.captured_representations["token_hidden"][layer].float()
                        state=mblock.captured_representations["ssm_state_summary"][layer].flatten(1)
                        if z.shape[:2]!=mh.shape[:2] or z.shape[:2]!=valid.shape:
                            raise RuntimeError(
                                "Minimal/Mamba token alignment mismatch: "
                                f"stage={stage} layer={layer} minimal={tuple(z.shape)} "
                                f"mamba={tuple(mh.shape)} valid={tuple(valid.shape)}"
                            )
                        positions=torch.linspace(-1,1,z.shape[1],device=z.device)[None,:,None].expand(z.shape[0],-1,-1)
                        state=state[:,None,:].expand(-1,z.shape[1],-1)
                        mask=valid
                        b=buckets[stage][layer]
                        for name,value in (("z",z),("mamba_hidden",mh),("state",state),
                                           ("position",positions),("residual",residual)):
                            b[name].append(value[mask].cpu())
                        count=int(mask.sum()); b["trajectory"].append(torch.full((count,),trajectory))
                        b["chunk"].append(torch.full((count,),chunk)); b["diffusion"].append(torch.full((count,),diffusion))
                        b["sequence"].append(sequence[mask].cpu())
                        b["token_position"].append(token_position[mask].cpu())
            if trajectory==0 or (trajectory+1)%8==0:
                print(json.dumps({"event":"representation_progress","manifest":manifest,
                                  "trajectories_done":trajectory+1}),flush=True)
    for capture in captures.values(): capture.close()
    _reset(raw)
    return {stage:{layer:{name:torch.cat(parts) for name,parts in bucket.items()}
                   for layer,bucket in layers_map.items()} for stage,layers_map in buckets.items()}


def default_bucket():
    return {name:[] for name in ("z","mamba_hidden","state","position","residual",
                                 "trajectory","chunk","diffusion","sequence","token_position")}


def main():
    args=_args(); device=torch.device(args.device); torch.cuda.set_device(device)
    sys.path.insert(0,str(Path(args.ardy_root).resolve()))
    from ardy.model import load_model
    from ardy_adapter import ArdySequenceKernelConfig,install_ardy_sequence_kernel
    model=load_model("Ardy-Core-RP-20FPS-Horizon40",device=args.device,text_encoder=False,
                     checkpoints_dir=args.checkpoints_dir)
    install_ardy_sequence_kernel(model,ArdySequenceKernelConfig(mode="minimal",attention_backend="sdpa",
                                    window_size_chunks=4,num_sink_tokens=4))
    raw=model.denoiser.model; a=torch.load(args.minimal_checkpoint,map_location=device,weights_only=False)
    raw.root_model.load_state_dict(a["root_model"],strict=True); raw.body_model.load_state_dict(a["body_model"],strict=True)
    raw.eval(); mroot,mbody=_make_blocks(raw,8,device)
    m=torch.load(args.mamba_checkpoint,map_location=device,weights_only=False)
    mroot.load_state_dict(m["root_model"],strict=True); mbody.load_state_dict(m["body_model"],strict=True)
    mroot.eval(); mbody.eval()
    result={"train":_collect(raw,mroot,mbody,args.train_manifest,device,args.layers),
            "development":_collect(raw,mroot,mbody,args.dev_manifest,device,args.layers),
            "layers":args.layers,"final_heldout_used":False}
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); torch.save(result,out/"representations.pt")
    summary={split:{stage:{str(layer):len(tensors["z"]) for layer,tensors in ls.items()}
                    for stage,ls in stages.items()} for split,stages in result.items() if split in {"train","development"}}
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n"); print(json.dumps(summary,indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
