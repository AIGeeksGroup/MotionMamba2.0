"""Build the preregistered actor-aware D3b 20 s selection panel."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np

GOAL_JOINTS = (0, 6, 13, 24, 32)

def sha(path: Path) -> str:
    h = hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True); p.add_argument("--text-cache", required=True)
    p.add_argument("--output", required=True); p.add_argument("--audit", required=True)
    p.add_argument("--samples", type=int, default=32); p.add_argument("--seed", type=int, default=20260906)
    a = p.parse_args(); manifest = Path(a.manifest); rows = [json.loads(x) for x in manifest.read_text().splitlines() if x.strip()]
    eligible = [r for r in rows if int(r["frames"]) >= 500]
    by_actor = {}
    for r in eligible: by_actor.setdefault(str(r["actor_uid"]), []).append(r)
    for values in by_actor.values(): values.sort(key=lambda r: (str(r.get("source_id", r["id"])), str(r["id"])))
    actors = sorted(by_actor); rng = np.random.default_rng(a.seed); rng.shuffle(actors)
    selected=[]; pos={x:0 for x in actors}
    while len(selected) < a.samples:
        progress=False
        for actor in actors:
            if pos[actor] < len(by_actor[actor]):
                selected.append(by_actor[actor][pos[actor]]); pos[actor]+=1; progress=True
                if len(selected) == a.samples: break
        if not progress: break
    if len(selected) < a.samples: raise RuntimeError(f"only {len(selected)} eligible rows")
    index = json.loads((Path(a.text_cache)/"index.json").read_text())
    captions = {str(x["id"]): str(x["caption"]) for x in index["samples"]}
    out=[]
    for offset,r in enumerate(selected):
        aug_seed=a.seed+1000003*offset; rng=np.random.default_rng(aug_seed)
        start=int(rng.integers(int(r["frames"])-500+1)); pid=f"d3b20s:{r['id']}:{start}:500:in_horizon"
        goals=list(range(52-1, 476, 52));
        if not goals or goals[-1] != 475: goals.append(475)
        out.append({"panel_id":pid,"id":r["id"],"actor_uid":r["actor_uid"],"source_id":r.get("source_id",r["id"]),"feature_path":r["feature_path"],"source_path":r.get("source_path"),"caption":captions[r["id"]],"augmentation_seed":aug_seed,"noise_seed":a.seed*1000003+offset,"crop_start":start,"crop_end":start+500,"selection_span_frames":500,"total_frames":500,"history_frames":24,"horizon_frames":52,"protocol":"in_horizon","goal_frames":goals,"goal_joints":[GOAL_JOINTS[int.from_bytes(hashlib.sha256(f"{aug_seed}:{pid}:goal:{i}".encode()).digest()[:8],"little")%len(GOAL_JOINTS)] for i in range(len(goals))],"source_manifest_sha256":sha(manifest),"split_role":"d3b_selection"})
    output=Path(a.output); output.parent.mkdir(parents=True,exist_ok=True); output.write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in out))
    audit={"schema_version":"d3b_20s_panel_eligibility_v1","manifest":str(manifest),"manifest_sha256":sha(manifest),"text_cache":str(Path(a.text_cache)),"seed":a.seed,"minimum_frames":500,"eligible_records":len(eligible),"eligible_actors":len(by_actor),"selected_motions":len(out),"selected_actors":len({str(x['actor_uid']) for x in out}),"selected_ids":[x["id"] for x in out],"selection_rule":"deterministic actor-round-robin over all records with >=500 frames; no locked-final data"}
    Path(a.audit).write_text(json.dumps(audit,indent=2)+"\n")

if __name__ == "__main__": main()
