"""Download the gated BONES-SEED dataset with resumable Hugging Face Hub support."""

from __future__ import annotations

import os

from huggingface_hub import snapshot_download


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not available in the environment")
    print("download_start", flush=True)
    path = snapshot_download(
        repo_id="bones-studio/seed",
        repo_type="dataset",
        local_dir="/data4/zts/datasets/bones_seed",
        token=token,
        endpoint="https://huggingface.co",
        max_workers=3,
    )
    print("download_complete", path, flush=True)


if __name__ == "__main__":
    main()
