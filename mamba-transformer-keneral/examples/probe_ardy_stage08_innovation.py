"""Fit frozen state predictors and parameter-budgeted P0--P5 probes."""

from __future__ import annotations

import argparse,json,math,random
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn


def args():
    p=argparse.ArgumentParser(); p.add_argument("--representations",required=True)
    p.add_argument("--output-dir",required=True); p.add_argument("--steps",type=int,default=500)
    p.add_argument("--batch-size",type=int,default=4096); p.add_argument("--seed",type=int,default=149)
    p.add_argument("--device",default="cuda:0"); return p.parse_args()


class MLP(nn.Module):
    def __init__(self,din,dout,width): super().__init__(); self.net=nn.Sequential(nn.Linear(din,width),nn.SiLU(),nn.Linear(width,dout))
    def forward(self,x): return self.net(x)


class GenericGate(nn.Module):
    def __init__(self,dz,ds,dout,width=64):
        super().__init__(); self.dz=dz; self.z=nn.Linear(dz,width); self.s=nn.Linear(ds,width)
        self.gate=nn.Linear(2*width,width); self.out=nn.Linear(width,dout)
    def forward(self,x):
        z=torch.tanh(self.z(x[:,:self.dz])); s=torch.tanh(self.s(x[:,self.dz:])); g=torch.sigmoid(self.gate(torch.cat((z,s),-1)))
        return self.out(g*z+(1-g)*s)


def fit(model,x,y,xd,yd,steps,batch,seed):
    model.train(); opt=torch.optim.AdamW(model.parameters(),lr=1e-3); g=torch.Generator(device=x.device).manual_seed(seed)
    first=None
    for _ in range(steps):
        idx=torch.randint(len(x),(min(batch,len(x)),),generator=g,device=x.device)
        loss=F.mse_loss(model(x[idx]),y[idx]); first=float(loss) if first is None else first
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad(): pred=model(xd); mse=float(F.mse_loss(pred,yd)); base=float(F.mse_loss(yd,yd.mean(0,keepdim=True).expand_as(yd)))
    return {"initial_batch_mse":first,"development_mse":mse,"r2":1-mse/max(base,1e-12),
            "parameters":sum(p.numel() for p in model.parameters())},pred


def normalize(x,mean=None,std=None):
    if mean is None: mean=x.mean(0,keepdim=True); std=x.std(0,keepdim=True).clamp_min(1e-4)
    return (x-mean)/std,mean,std


def source(tensors,name):
    if name=="hidden": return tensors["mamba_hidden"]
    return torch.cat((tensors["state"],tensors["position"]),dim=-1)


def width_for(din,dout,budget=70000): return max(8,int((budget-dout)/(din+dout+1)))


def gate_width_for(dz,ds,dout,budget=70000):
    """Largest GenericGate width that does not exceed the probe budget."""
    def count(width):
        return (dz+1)*width+(ds+1)*width+(2*width+1)*width+(width+1)*dout
    widths=[width for width in range(8,257) if count(width)<=budget]
    return max(widths) if widths else 8


def main():
    a=args(); random.seed(a.seed); torch.manual_seed(a.seed); device=torch.device(a.device); torch.cuda.set_device(device)
    data=torch.load(a.representations,map_location="cpu",weights_only=False); predictors={}; candidates=[]
    for stage in ("root","body"):
      for layer in data["layers"]:
       for state_name in ("hidden","recurrent"):
        tr=data["train"][stage][layer]; dv=data["development"][stage][layer]
        x=source(tr,state_name).float().to(device); xd=source(dv,state_name).float().to(device)
        x,m,s=normalize(x); xd,_,_=normalize(xd,m,s)
        z=tr["z"].float().to(device); zd=dv["z"].float().to(device)
        model=MLP(x.shape[1],z.shape[1],256).to(device)
        metric,pred=fit(model,x,z,xd,zd,a.steps,a.batch_size,a.seed+layer)
        with torch.no_grad(): metric["cosine"]=float(F.cosine_similarity(pred,zd,dim=-1).mean())
        key=f"{stage}:L{layer}:{state_name}"; predictors[key]=metric
        print(f"predictor {key}: dev_mse={metric['development_mse']:.8f} "
              f"r2={metric['r2']:.5f} cosine={metric['cosine']:.5f}", flush=True)
        candidates.append((metric["development_mse"],stage,layer,state_name,model.state_dict(),m.cpu(),s.cpu()))
    selected={}
    for stage in ("root","body"):
        selected[stage]=min((c for c in candidates if c[1]==stage),key=lambda c:c[0])
    probes={}; innovation_stats={}
    for stage,choice in selected.items():
        _,_,layer,state_name,state_dict,mean,std=choice; tr=data["train"][stage][layer]; dv=data["development"][stage][layer]
        print(f"selected {stage}: layer={layer} source={state_name}", flush=True)
        st=source(tr,state_name).float().to(device); sd=source(dv,state_name).float().to(device)
        st=(st-mean.to(device))/std.to(device); sd=(sd-mean.to(device))/std.to(device)
        z=tr["z"].float().to(device); zd=dv["z"].float().to(device)
        predictor=MLP(st.shape[1],z.shape[1],256).to(device); predictor.load_state_dict(state_dict); predictor.eval()
        with torch.no_grad(): zh=predictor(st); zhd=predictor(sd); delta=F.layer_norm(z,z.shape[-1:])-F.layer_norm(zh,zh.shape[-1:]); deltad=F.layer_norm(zd,zd.shape[-1:])-F.layer_norm(zhd,zhd.shape[-1:])
        residual=tr["residual"].float().to(device); residuald=dv["residual"].float().to(device)
        innovation_stats[stage]={"layer":layer,"state_source":state_name,
            "z_norm":float(zd.norm(dim=-1).mean()),"zhat_norm":float(zhd.norm(dim=-1).mean()),
            "delta_norm":float(deltad.norm(dim=-1).mean()),
            "delta_z_ratio":float(deltad.norm(dim=-1).mean()/zd.norm(dim=-1).mean().clamp_min(1e-12))}
        feature_map={"P0_z":(z,zd),"P1_state":(st,sd),"P2_concat":(torch.cat((z,st),-1),torch.cat((zd,sd),-1)),
                     "P3_delta":(delta,deltad),"P4_zhat_delta":(torch.cat((zh,delta),-1),torch.cat((zhd,deltad),-1)),
                     "P5_generic_gate":(torch.cat((z,st),-1),torch.cat((zd,sd),-1))}
        probes[stage]={}
        for index,(name,(x,xd)) in enumerate(feature_map.items()):
            # Every probe receives features standardized with TRAIN statistics.
            # This prevents the normalized innovation from receiving an unfair
            # optimization advantage over raw z/state concatenations.
            x,feature_mean,feature_std=normalize(x)
            xd,_,_=normalize(xd,feature_mean,feature_std)
            model=(GenericGate(z.shape[1],st.shape[1],residual.shape[1],
                               gate_width_for(z.shape[1],st.shape[1],residual.shape[1])).to(device)
                   if name=="P5_generic_gate" else
                   MLP(x.shape[1],residual.shape[1],width_for(x.shape[1],residual.shape[1])).to(device))
            metric,_=fit(model,x,residual,xd,residuald,a.steps,a.batch_size,a.seed+100+index)
            probes[stage][name]=metric
            print(f"probe {stage}/{name}: dev_mse={metric['development_mse']:.8f} "
                  f"r2={metric['r2']:.5f} params={metric['parameters']}", flush=True)
    result={"final_heldout_used":False,"state_predictors":predictors,
            "selected":{"root":{"layer":selected["root"][2],"source":selected["root"][3]},
                        "body":{"layer":selected["body"][2],"source":selected["body"][3]}},
            "innovation_statistics":innovation_stats,"probes":probes}
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); (out/"probe_results.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
