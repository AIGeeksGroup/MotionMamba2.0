"""Positive-control output bridge trained only from frozen parent predictions."""

from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn

from analyze_ardy_stage07_complementarity import _evaluate,_fit_affine,_fit_gate,_fit_scalar
from evaluate_ardy_attention_mamba_complementarity import _combined


def main():
    p=argparse.ArgumentParser(); p.add_argument("--records",required=True); p.add_argument("--output-dir",required=True); p.add_argument("--seed",type=int,default=157)
    a=p.parse_args(); torch.manual_seed(a.seed); cached=torch.load(a.records,map_location="cpu",weights_only=False)
    train=_combined(cached["train"]); dev=_combined(cached["development"])
    scalar=_fit_scalar(train); affine=_fit_affine(train); gate=_fit_gate(train,a.seed)
    # Exact plumbing identity for F0.
    fixed=train[0]; expected=.5*(fixed["minimal"]+fixed["mamba"]); implemented=fixed["minimal"]+.5*(fixed["mamba"]-fixed["minimal"])
    f0_error=float((expected-implemented).abs().max())
    # Per-output gate is an intentionally high-capacity one-case positive
    # control. It tests alignment and objective plumbing, not generalization.
    logit=nn.Parameter(torch.zeros_like(fixed["minimal"])); opt=torch.optim.AdamW([logit],lr=.1)
    initial=float(F.mse_loss(fixed["minimal"],fixed["target"])); first=None
    for _ in range(500):
        prediction=fixed["minimal"]+torch.sigmoid(logit)*(fixed["mamba"]-fixed["minimal"])
        loss=F.mse_loss(prediction,fixed["target"]); first=float(loss) if first is None else first
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    final=float(F.mse_loss(fixed["minimal"]+torch.sigmoid(logit)*(fixed["mamba"]-fixed["minimal"]),fixed["target"]))
    result={"final_heldout_used":False,"fixed_average_max_abs_error":f0_error,
            "tiny":{"minimal_mse":initial,"initial_average_mse":first,"final_gate_mse":final,
                    "reduction_vs_minimal":(initial-final)/initial},
            "fitted_scalar_weight_on_minimal":scalar,"fitted_affine":affine.tolist(),
            "train":_evaluate(train,scalar,affine,gate),"development":_evaluate(dev,scalar,affine,gate)}
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); (out/"output_bridge.json").write_text(json.dumps(result,indent=2)+"\n")
    torch.save({"gate":gate.state_dict(),"tiny_logit":logit.detach(),"result":result},out/"output_bridge.pt")
    print(json.dumps(result,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
