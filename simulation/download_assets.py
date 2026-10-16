"""Fetch pinned runtime assets; no historical training data is required."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "Royalman1/Motion-mamba-2.0"
WEIGHT = "moiton mamba 2.0.pt"
WEIGHT_SHA = "2e828449addff3aa079387f319589cf4ac9b202c48577278a6d274f28b14f679"
MODELS = [
    ("nvidia/ARDY-G1-RP-25FPS-Horizon52", "059b8007df0ba194a006a877b59a563955ac7b70", "official/ARDY-G1-RP-25FPS-Horizon52"),
    ("meta-llama/Meta-Llama-3-8B-Instruct", "8afb486c1db24fe5011ec46dfbe5b5dccdb575c2", None),
    ("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp", "31474e395ada192e8ed1586db6be79fb3b70c9c0", "text_encoders/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"),
    ("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised", "baa8ebf04a1c2500e61288e7dad65e8ae42601a7", "text_encoders/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised"),
]
SONIC = {
    "model_encoder.onnx": ("policy/release/model_encoder.onnx", "013ab0287236aa2721e13f1e936d699db982302d0de0bfcdae76d5c3245362d3"),
    "model_decoder.onnx": ("policy/release/model_decoder.onnx", "c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed"),
    "observation_config.yaml": ("policy/release/observation_config.yaml", "466d05947c78af6c76388adfb86e3a2a77b2a1d921a64883ed3d085ebf58de1b"),
    "planner_sonic.onnx": ("planner/target_vel/V2/planner_sonic.onnx", "39b553e197f62f077975ba38512bc04781a3fc37c2af7c6756e04629f760edea"),
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-checkpoint", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("HF_HOME", str(ROOT / "cache/huggingface"))
    cache = str(ROOT / "cache/huggingface/hub")
    records = []
    if not args.skip_checkpoint:
        source = hf_hub_download(REPOSITORY, WEIGHT, cache_dir=cache)
        target = ROOT / "checkpoints" / WEIGHT
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        assert sha(target) == WEIGHT_SHA, "Downloaded checkpoint hash mismatch"
    for repo, revision, relative in MODELS:
        snapshot = Path(snapshot_download(repo, revision=revision, cache_dir=cache,
            ignore_patterns=["*.trt", "*.engine", "*.bin", "*.pth", "original/*"]))
        # LLM2Vec's adapter config references the original Meta model ID. Keep
        # that pinned snapshot in the same Hub cache for offline resolution.
        if relative:
            target = ROOT / "checkpoints" / relative
            shutil.copytree(snapshot, target, dirs_exist_ok=True)
        if repo != "nvidia/ARDY-G1-RP-25FPS-Horizon52":
            # The unmodified LLM2Vec loader names these models without a
            # revision. A commit-pinned snapshot alone does not create refs/main;
            # register the pinned ref explicitly for its later offline lookup.
            ref = Path(cache)/("models--" + repo.replace("/", "--"))/"refs/main"
            ref.parent.mkdir(parents=True, exist_ok=True)
            ref.write_text(revision)
        records.append(dict(repo=repo, revision=revision))
    deploy = ROOT / "GR00T-WholeBodyControl/gear_sonic_deploy"
    for filename, (relative, expected) in SONIC.items():
        source = hf_hub_download("nvidia/GEAR-SONIC", filename,
            revision="5e22ddc69abcea2a9aafc40536b14c232d3f9d7f", cache_dir=cache)
        if sha(source) != expected: raise RuntimeError("SONIC asset hash mismatch: " + filename)
        target = deploy / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    target = ROOT / "runtime/ASSET_DOWNLOAD.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(models=records, sonic=SONIC, checkpoint_sha256=WEIGHT_SHA), indent=2) + "\n")
    print("Pinned runtime assets ready:", target)


if __name__ == "__main__": main()
