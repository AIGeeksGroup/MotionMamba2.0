"""Analyze D3 intervention outputs with paired actor-cluster bootstrap."""
from __future__ import annotations
import argparse, hashlib, json, subprocess
from pathlib import Path
import numpy as np
import torch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from long_memory_protocol import paired_cluster_bootstrap, window_summary

METRICS = (
    "root_post_w5_auc_cm_windows", "constraint_post_w5_auc_cm_windows",
    "root_post_w5_slope_cm_per_window", "constraint_post_w5_slope_cm_per_window",
    "root_final_window_cm", "constraint_final_window_cm",
)

def rows(path: Path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]

def paired(base: Path, cand: Path, seed: int):
    left, right = rows(base), rows(cand)
    if [x["panel_id"] for x in left] != [x["panel_id"] for x in right]:
        raise ValueError("panel ordering differs")
    out = {}
    for metric in METRICS:
        pairs = [{"actor_uid": a["actor_uid"], "id": a["motion_id"],
                  "difference": float(b[metric]) - float(a[metric])}
                 for a, b in zip(left, right)]
        out[metric] = paired_cluster_bootstrap(pairs, value_key="difference",
                                                resamples=20_000, seed=seed)
    return out

def sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser(); p.add_argument("--root", required=True)
    p.add_argument("--output", required=True); p.add_argument("--panel-manifest", required=True)
    a = p.parse_args(); root = Path(a.root); normal = root / "normal/40s_in_horizon/per_sample.jsonl"
    aliases = {"one_shot_permute_memory":"one_shot_permute_memory", "persistent_read_permutation":"persistent_read_permutation",
               "freeze_write_after_w4":"freeze_write_after_w4", "one_shot_permute_plus_freeze_write":"one_shot_permute_plus_freeze_write",
               "semantic_mismatch_persistent_read_permutation":"semantic_mismatch_persistent_read_permutation",
               "write_off":"write_off", "readout_off":"readout_off"}
    comparisons = {name: paired(normal, root / f"{directory}/40s_in_horizon/per_sample.jsonl", 20260905 + i)
                   for i, (name, directory) in enumerate(aliases.items())}
    normal_result = json.loads((root / "normal/40s_in_horizon/result.json").read_text())
    checkpoint = Path(normal_result["checkpoint"])
    repo = Path(__file__).resolve().parents[1]
    source_paths = [repo / "src/ardy_adapter/diffusion_memory.py", repo / "src/ardy_adapter/ardy_dit_block.py", repo / "src/sequence_kernel/operator.py", repo / "examples/evaluate_ardy_official_table5.py"]
    results = {"schema_version":"d3_intervention_analysis_v1", "git_sha":subprocess.check_output(["git","rev-parse","HEAD"], cwd=repo, text=True).strip(),
               "git_worktree_dirty":bool(subprocess.check_output(["git","status","--porcelain"],cwd=repo,text=True).strip()),"source_sha256":{str(p.relative_to(repo)):sha(p) for p in source_paths},
               "checkpoint":str(checkpoint.resolve()), "checkpoint_sha256":sha(checkpoint), "panel_manifest":str(Path(a.panel_manifest).resolve()),
               "protocol":{"sample_ids":normal_result["sample_ids"],"actor_uids":normal_result["actor_uids"],"panel_sha256":normal_result["panel_sha256"],"diffusion_steps":normal_result["diffusion_steps"],"cfg_weight_text":normal_result["cfg_weight_text"],"cfg_weight_constraint":normal_result["cfg_weight_constraint"],"attention_window_chunks":normal_result["attention_window_chunks"]},
               "runtime":{"torch":torch.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
               "bootstrap":{"resamples":20000,"seed_base":20260905,"unit":"actor_then_motion_within_actor"},
               "comparisons":comparisons}
    Path(a.output).write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))

if __name__ == "__main__": main()
